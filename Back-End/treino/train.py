"""Treina o avaliador de redações (BERTimbau ajustado no Essay-BR).

Padrão: 512 tokens, BERT inteiro, 4 épocas, tema misto e aumento de dados contra
vieses (modelo v3). Grava em saved_model/ os pesos, o tokenizador, config.json e
results.json. Depois, rode a calibração dos cortes (treino/calibrate.py).

Uso (dentro de Back-End):  python -m treino.train [opções]   (veja --help)
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # permite rodar também como "python treino/train.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from avaliador.caminhos import PASTA_MODELO  # noqa: E402
from avaliador.dados import TIPOS_DE_DIVISAO, TIPOS_DE_TEMA, load_splits  # noqa: E402
from avaliador.modelo import MAX_LEN, MODEL_NAME, EnemDataset, EnemEssayEvaluator  # noqa: E402
from avaliador.notas import post_process_scores, qwk  # noqa: E402
from avaliador.perturbacoes import aumentar_contra_vies  # noqa: E402


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, use_amp=False):
    """Perda e QWK por competência (arredondamento simples) num DataLoader."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(ids, mask)
            preds = preds.float()
            total_loss += criterion(preds, labels).item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)

    y_pred_rounded = post_process_scores(preds_arr)
    y_true_rounded = np.round(targets_arr * 200.0).astype(int)

    qwk_scores = [qwk(y_true_rounded[:, i], y_pred_rounded[:, i]) for i in range(5)]
    return avg_loss, qwk_scores


def parse_args():
    p = argparse.ArgumentParser(description="Treina o avaliador de redações (BERTimbau).")
    p.add_argument("--max-len", type=int, default=MAX_LEN, help="tokens por redação (máx. 512)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1, help="passos acumulados por atualização")
    p.add_argument("--frozen-layers", type=int, default=0,
                   help="quantas camadas iniciais do BERT ficam congeladas (0 = treina todas)")
    p.add_argument("--warmup", type=float, default=0.1,
                   help="fração dos passos com aquecimento da taxa de aprendizado (0 = sem agendador)")
    p.add_argument("--select-by", choices=["loss", "qwk"], default="qwk",
                   help="critério para escolher o melhor checkpoint na validação")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="desliga a precisão mista bf16 (ligada por padrão na GPU)")
    p.add_argument("--tema", choices=TIPOS_DE_TEMA, default="misto",
                   help="tema usado na entrada: misto (padrão) sorteia trecho do texto motivador "
                        "ou título curto, para o modelo funcionar com o título que o usuário digita")
    p.add_argument("--aumento-vies", type=float, default=0.5,
                   help="fração das redações de treino que ganham uma cópia alterada contra os vieses "
                        "(avaliador/perturbacoes.py); padrão 0.5 (modelo v3), 0 = desligado (modelo v2)")
    p.add_argument("--divisao", choices=TIPOS_DE_DIVISAO, default="aleatoria",
                   help="aleatoria (padrão) ou por_tema (temas de validação e teste fora do treino)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(PASTA_MODELO), help="pasta onde o modelo é salvo (padrão: saved_model)")
    p.add_argument("--no-test", action="store_true",
                   help="não avalia no conjunto de teste (use durante os experimentos)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Configuração: {vars(args)}", flush=True)

    splits = load_splits(tema=args.tema, divisao=args.divisao)
    X_train_t, X_train_txt, y_train = splits["train"]
    X_val_t, X_val_txt, y_val = splits["val"]
    X_test_t, X_test_txt, y_test = splits["test"]

    # Aumento de dados contra os vieses: só no TREINO (validação e teste ficam intactos)
    aumentos = {}
    if args.aumento_vies > 0:
        n_antes = len(X_train_txt)
        X_train_t, X_train_txt, y_train, aumentos = aumentar_contra_vies(
            X_train_t, X_train_txt, y_train, args.aumento_vies, seed=args.seed
        )
        print(f"Aumento contra vieses: {n_antes} -> {len(X_train_txt)} redações de treino {aumentos}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = EnemDataset(X_train_txt, X_train_t, y_train, tokenizer, args.max_len)
    val_dataset = EnemDataset(X_val_txt, X_val_t, y_val, tokenizer, args.max_len)
    test_dataset = EnemDataset(X_test_txt, X_test_t, y_test, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = EnemEssayEvaluator(model_name=MODEL_NAME).to(device)

    if args.frozen_layers > 0:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[: args.frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    criterion = nn.MSELoss()

    total_batches = len(train_loader)
    updates_per_epoch = max(1, total_batches // args.grad_accum)
    scheduler = None
    if args.warmup > 0:
        total_updates = updates_per_epoch * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(args.warmup * total_updates), total_updates
        )

    # Treinamento e salvamento do melhor checkpoint
    log_every = 50
    best_val_loss = float("inf")
    best_val_qwk = -float("inf")
    best_epoch = 0
    history = []
    os.makedirs(args.out, exist_ok=True)
    weights_path = os.path.join(args.out, "model_weights.pt")

    train_start_time = time.perf_counter()

    print(
        f"Iniciando treinamento em: {device} | Treino: {len(train_dataset)} ({total_batches} batches) | "
        f"Val: {len(val_dataset)} | Teste: {len(test_dataset)}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.perf_counter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                preds = model(ids, mask)
            loss = criterion(preds.float(), labels)
            (loss / args.grad_accum).backward()
            train_loss += loss.item()

            if step % args.grad_accum == 0 or step == total_batches:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            if step % log_every == 0 or step == total_batches:
                elapsed_epoch = time.perf_counter() - epoch_start
                eta_epoch = elapsed_epoch / step * (total_batches - step)
                print(
                    f"  [Época {epoch + 1}/{args.epochs}] Batch {step}/{total_batches} "
                    f"({100.0 * step / total_batches:.1f}%) | "
                    f"Loss: {train_loss / step:.4f} | "
                    f"Decorrido: {format_duration(elapsed_epoch)} | "
                    f"ETA Época: {format_duration(eta_epoch)}",
                    flush=True,
                )

        avg_val_loss, qwk_scores = evaluate(model, val_loader, criterion, device, use_amp)
        val_qwk = float(np.mean(qwk_scores))
        epoch_duration = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss / total_batches, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_qwk": round(val_qwk, 4),
            "val_qwk_por_competencia": [round(s, 4) for s in qwk_scores],
        })

        print(
            f"Época {epoch + 1}/{args.epochs} concluída em {format_duration(epoch_duration)} | "
            f"Train Loss: {train_loss / total_batches:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Tempo total: {format_duration(time.perf_counter() - train_start_time)}",
            flush=True,
        )
        print(
            f"QWK Médio Validação: {val_qwk:.4f} "
            f"(C1-C5: {[round(s, 4) for s in qwk_scores]})",
            flush=True,
        )

        improved = (
            val_qwk > best_val_qwk if args.select_by == "qwk" else avg_val_loss < best_val_loss
        )
        if improved:
            best_val_loss = min(best_val_loss, avg_val_loss)
            best_val_qwk = val_qwk
            best_epoch = epoch + 1
            torch.save(model.state_dict(), weights_path)
            tokenizer.save_pretrained(os.path.join(args.out, "tokenizer"))
            print(f"-> Novo melhor modelo salvo em {args.out}", flush=True)
        best_val_loss = min(best_val_loss, avg_val_loss)

    total_time = time.perf_counter() - train_start_time
    results = {
        "config": vars(args),
        "max_len": args.max_len,
        "model_name": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_val_qwk": round(best_val_qwk, 4),
        "history": history,
        "train_time_s": round(total_time, 1),
    }

    # Avaliação do melhor checkpoint no conjunto de teste
    if not args.no_test:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        test_loss, test_qwk = evaluate(model, test_loader, criterion, device, use_amp)
        results["test_loss"] = round(test_loss, 4)
        results["test_qwk"] = round(float(np.mean(test_qwk)), 4)
        results["test_qwk_por_competencia"] = [round(s, 4) for s in test_qwk]
        print("\n--- RESULTADO FINAL NO CONJUNTO DE TESTE ---", flush=True)
        print(f"Melhor época (validação): {best_epoch}", flush=True)
        print(f"Tempo total de execução: {format_duration(total_time)}", flush=True)
        print(f"Test Loss (MSE): {test_loss:.4f}", flush=True)
        print(
            f"QWK Médio Teste: {np.mean(test_qwk):.4f} "
            f"(C1-C5: {[round(s, 4) for s in test_qwk]})",
            flush=True,
        )
    else:
        print(f"\nMelhor época: {best_epoch} | QWK médio validação: {best_val_qwk:.4f}", flush=True)

    # config.json: lido pela API para usar o mesmo max_len do treino
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"max_len": args.max_len, "model_name": MODEL_NAME, "tema_treino": args.tema, "divisao": args.divisao,
                   "aumento_vies": args.aumento_vies}, f, indent=2)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

import { Link } from "@tanstack/react-router";
import { ServerSettings } from "./server-settings";

const LINKS = [
  { to: "/", label: "Corrigir" },
  { to: "/exemplos", label: "Exemplos" },
  { to: "/dicas", label: "Dicas" },
  { to: "/sobre", label: "O modelo" },
] as const;

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-2 sm:h-16 sm:flex-nowrap sm:px-6 sm:py-0">
        <Link to="/" className="group flex items-baseline gap-2">
          <span className="font-display text-2xl font-semibold tracking-tight text-ink">Nota</span>
          <span className="relative font-display text-2xl font-semibold italic text-pen">
            Mil
            <svg
              aria-hidden
              viewBox="0 0 60 30"
              className="pointer-events-none absolute -left-2 -top-1.5 h-[2.6rem] w-[3.4rem] text-pen/70"
              fill="none"
            >
              <path
                d="M8 16c2-9 18-13 32-11 11 2 17 8 12 15-6 8-28 9-40 5C4 22 3 15 12 10"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </span>
        </Link>
        {/* Em telas estreitas o menu desce para uma segunda linha, com rolagem lateral se precisar */}
        <nav className="-mx-2 flex w-full items-center justify-between gap-0.5 overflow-x-auto text-sm sm:mx-0 sm:w-auto sm:justify-end sm:gap-1">
          {LINKS.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className="whitespace-nowrap rounded-md px-2 py-1.5 font-medium text-muted-foreground transition-colors hover:text-ink sm:px-3"
              activeProps={{ className: "text-ink underline decoration-pen decoration-2 underline-offset-8" }}
              activeOptions={{ exact: true }}
            >
              {l.label}
            </Link>
          ))}
          <ServerSettings />
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>
          <span className="font-display font-semibold text-ink">Nota Mil</span> · correção automática
          de redações no modelo ENEM
        </p>
        <p>As notas são uma estimativa da rede neural e não substituem um corretor humano.</p>
      </div>
    </footer>
  );
}

import { useEffect, useState } from "react";
import { DEFAULT_API_URL, getApiUrl, onApiUrlChange } from "@/lib/api";

export function useApiUrl() {
  const [url, setUrl] = useState(DEFAULT_API_URL);
  useEffect(() => {
    setUrl(getApiUrl());
    return onApiUrlChange(() => setUrl(getApiUrl()));
  }, []);
  return url;
}

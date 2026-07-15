const ALPHATAB_VERSION = "1.8.1";
export const ALPHATAB_CDN = `https://cdn.jsdelivr.net/npm/@coderline/alphatab@${ALPHATAB_VERSION}/dist`;

export type AlphaTabModule = typeof import("@coderline/alphatab");

declare global {
  interface Window {
    alphaTab?: AlphaTabModule;
  }
}

let loadPromise: Promise<AlphaTabModule> | null = null;

/** Load alphaTab from CDN to avoid Next.js/webpack worker bundling issues. */
export function loadAlphaTab(): Promise<AlphaTabModule> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("alphaTab can only load in the browser"));
  }
  if (window.alphaTab?.AlphaTabApi) {
    return Promise.resolve(window.alphaTab);
  }
  if (loadPromise) return loadPromise;

  loadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-alphatab="true"]');
    if (existing) {
      existing.addEventListener("load", () => {
        if (window.alphaTab?.AlphaTabApi) resolve(window.alphaTab);
        else reject(new Error("alphaTab global missing after script load"));
      });
      existing.addEventListener("error", () => reject(new Error("alphaTab script failed")));
      return;
    }

    const script = document.createElement("script");
    script.src = `${ALPHATAB_CDN}/alphaTab.min.js`;
    script.async = true;
    script.dataset.alphatab = "true";
    script.onload = () => {
      if (window.alphaTab?.AlphaTabApi) resolve(window.alphaTab);
      else reject(new Error("alphaTab global missing after script load"));
    };
    script.onerror = () => reject(new Error("alphaTab script failed to load"));
    document.head.appendChild(script);
  });

  return loadPromise;
}

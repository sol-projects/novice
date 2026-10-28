import React, { createContext, useContext, useEffect, useState } from "react";
import "./appearance.css";
import "./palettes.css";

export const themes = [
  { id: "almanac", name: "Almanac", description: "Izvirni arhiv. Uredniški seznam, hladni papir in magenta.", detail: "Uredniški seznam" },
  { id: "apple", name: "Apple", description: "Mirna čitalnica. Stranska navigacija, mehke površine in veliko prostora.", detail: "Prostor za branje" },
  { id: "liquid-apple", name: "Liquid apple", description: "Novice skozi kristal. Lebdeča navigacija, ledeno steklo in mehki odsevi.", detail: "Kristalna čitalnica" },
  { id: "maximalist", name: "Maksimalizem", description: "Novice kot plakat. Veliki naslovi, drzen prelom in debeli robovi.", detail: "Glasna naslovnica" },
  { id: "terminal", name: "Hacker terminal", description: "Arhiv v konzoli. Kompaktne vrstice, fiksna širina pisave in zelena svetloba.", detail: "Podatkovna konzola" },
  { id: "kitty", name: "Hello Kitty", description: "Tvoj dnevnik novic. Rožnate kartice, mašnice in igrive podrobnosti.", detail: "Rožnati dnevnik" },
  { id: "garden", name: "Botanični atlas", description: "Topel papir, zelena tinta in časopisni stolpci. Novice za počasno branje.", detail: "Svetla časopisna izdaja" },
  { id: "hedgerow", name: "Nočni observatorij", description: "Globoka zelena, kremni naslovi in širok pregled novic. Mirna nočna izdaja.", detail: "Temni pregled virov" },
  { id: "retro", name: "HTML 90", description: "Splet, kot se ga spomnimo. Modre povezave, sivi robovi in klasična tipografija.", detail: "Domača stran iz devetdesetih" },
  { id: "newspaper", name: "Časopis", description: "Odpri časopis. Velika naslovnica, tiskarski stolpci, inicialke in topla barva papirja.", detail: "Novice kot tiskana izdaja" },
] as const;
export const schemes = [
  { id: "original", name: "Izvirna", description: "Barve izbrane teme" },
  { id: "light", name: "Svetla", description: "Svetla paleta izbranega dizajna" },
  { id: "dark", name: "Temna", description: "Temna paleta izbranega dizajna" },
  { id: "forest", name: "Gozd", description: "Žajbelj in temno zelena" },
] as const;
export type ThemeId = typeof themes[number]["id"];
export type SchemeId = typeof schemes[number]["id"];
type Appearance = { theme: ThemeId; scheme: SchemeId };
const defaults: Appearance = { theme: "almanac", scheme: "original" };
const storageKey = "novice-appearance";

export function readAppearance(): Appearance {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || "null");
    return {
      theme: themes.some(theme => theme.id === value?.theme) ? value.theme : defaults.theme,
      scheme: schemes.some(scheme => scheme.id === value?.scheme) ? value.scheme : defaults.scheme,
    };
  } catch { return defaults; }
}

function loadTheme(theme: ThemeId): Promise<void> {
  if (theme === "almanac") return Promise.resolve();
  const id = `appearance-${theme}`;
  const existing = document.getElementById(id) as HTMLLinkElement | null;
  if (existing?.dataset.loaded) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const link = existing || document.createElement("link");
    const finish = (loaded: boolean) => {
      window.clearTimeout(timeout);
      link.removeEventListener("load", success);
      link.removeEventListener("error", failure);
      if (loaded) { link.dataset.loaded = "true"; resolve(); }
      else { link.remove(); reject(new Error("Theme stylesheet unavailable")); }
    };
    const success = () => finish(true);
    const failure = () => finish(false);
    const timeout = window.setTimeout(failure, 15000);
    link.addEventListener("load", success);
    link.addEventListener("error", failure);
    if (!existing) {
      link.id = id;
      link.rel = "stylesheet";
      link.href = `${process.env.PUBLIC_URL || ""}/themes/${theme}.css`;
      document.head.appendChild(link);
    }
  });
}

const AppearanceContext = createContext({ ...defaults, loading: false, message: "", change: async (_next: Appearance) => {} });
export const useAppearance = () => useContext(AppearanceContext);

export function AppearanceProvider({ children }: { children: React.ReactNode }) {
  const [appearance, setAppearance] = useState<Appearance>(defaults);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [target, setTarget] = useState<ThemeId>("almanac");

  useEffect(() => {
    let active = true;
    const saved = readAppearance();
    setTarget(saved.theme);
    setLoading(true);
    loadTheme(saved.theme).then(() => {
      if (!active) return;
      document.documentElement.dataset.theme = saved.theme;
      document.documentElement.dataset.scheme = saved.scheme;
      setAppearance(saved);
    }).catch(() => {
      if (active) setMessage("Shranjene teme ni bilo mogoče naložiti. Izberi jo znova.");
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const change = async (next: Appearance) => {
    if (loading) return;
    setTarget(next.theme);
    setLoading(true);
    setMessage("");
    try {
      await loadTheme(next.theme);
      document.documentElement.dataset.theme = next.theme;
      document.documentElement.dataset.scheme = next.scheme;
      setAppearance(next);
      try { localStorage.setItem(storageKey, JSON.stringify(next)); }
      catch { setMessage("Videz je nastavljen, vendar ga brskalnik ne more shraniti za naslednji obisk."); }
    } catch {
      setMessage("Teme ni bilo mogoče naložiti. Prejšnji videz ostaja; poskusi znova.");
    } finally { setLoading(false); }
  };

  return <AppearanceContext.Provider value={{ ...appearance, loading, message, change }}>
    {children}
    {loading && <div className="theme-loading" role="status"><span className="theme-spinner" aria-hidden="true" /><span>Pripravljamo videz<strong>{themes.find(theme => theme.id === target)!.name}</strong></span></div>}
  </AppearanceContext.Provider>;
}

import { useCallback, useEffect, useState } from "react";

const PREFIX = "mlprep.";

function read<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

/**
 * useState backed by localStorage. Values are namespaced under `mlprep.`
 * and kept in sync across tabs.
 */
export function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => read(key, initial));

  useEffect(() => {
    try {
      window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
    } catch {
      /* private mode / quota — progress just won't persist */
    }
  }, [key, value]);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key !== PREFIX + key || e.newValue === null) return;
      try {
        setValue(JSON.parse(e.newValue) as T);
      } catch {
        /* ignore malformed payloads from other tabs */
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [key]);

  const reset = useCallback(() => setValue(initial), [initial]);

  return [value, setValue, reset] as const;
}

/** A localStorage-backed set of string ids, exposed as a record for cheap JSON storage. */
export function useCheckedSet(key: string) {
  const [map, setMap] = useLocalStorage<Record<string, boolean>>(key, {});

  const has = useCallback((id: string) => Boolean(map[id]), [map]);

  const toggle = useCallback(
    (id: string) =>
      setMap((prev) => {
        const next = { ...prev };
        if (next[id]) delete next[id];
        else next[id] = true;
        return next;
      }),
    [setMap],
  );

  const set = useCallback(
    (id: string, on: boolean) =>
      setMap((prev) => {
        const next = { ...prev };
        if (on) next[id] = true;
        else delete next[id];
        return next;
      }),
    [setMap],
  );

  const clear = useCallback(() => setMap({}), [setMap]);

  const count = Object.keys(map).length;

  return { map, has, toggle, set, clear, count };
}

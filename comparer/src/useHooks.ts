import { useState } from "react";

export function useStateWithLocalStorageInt(initial: number, key: string): [number, (value: number) => void] {
  const [value, setValue] = useState<number>(parseInt(localStorage.getItem(key) || initial.toString()));
  const setValueAndStore = (newValue: number) => {
    localStorage.setItem(key, newValue.toString());
    setValue(newValue);
  }
  return [value, setValueAndStore];
}

export function useStateWithLocalStorageBoolean(initial: boolean, key: string): [boolean, (value: boolean) => void] {
  const ls = localStorage.getItem(key);
  const [value, setValue] = useState<boolean>(ls === "true" || ls === "false" ? ls === "true" : initial);
  const setValueAndStore = (newValue: boolean) => {
    localStorage.setItem(key, newValue.toString());
    setValue(newValue);
  }
  return [value, setValueAndStore];
}

export function useStateWithLocalStorageString(initial: string, key: string): [string, (value: string) => void] {
  const [value, setValue] = useState<string>(localStorage.getItem(key) ?? initial);
  const setValueAndStore = (newValue: string) => {
    localStorage.setItem(key, newValue.toString());
    setValue(newValue);
  }
  return [value, setValueAndStore];
}

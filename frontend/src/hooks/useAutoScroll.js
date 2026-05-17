import { useEffect, useRef } from "react";

export function useAutoScroll(deps) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return endRef;
}

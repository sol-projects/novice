import React, { useEffect, useRef } from "react";
import { useBreakpointValue } from "@chakra-ui/react";
import { FiSliders } from "react-icons/fi";
import Filter, { FilterProps } from "./Filter";

export default function FilterPanel({ onChange }: FilterProps) {
  const panel = useRef<HTMLDetailsElement>(null);
  const desktop = useBreakpointValue({ base: false, lg: true });

  useEffect(() => {
    if (panel.current) panel.current.open = !!desktop;
  }, [desktop]);

  return (
    <details ref={panel} className="filter-panel">
      <summary><FiSliders aria-hidden="true" />Filtri</summary>
      <Filter onChange={onChange} />
    </details>
  );
}

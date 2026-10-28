import React from "react";
import { useState, useEffect } from "react";
import DynamicInput from "./DynamicInput";

export interface FilterProps {
  onChange: (filterData: FilterData) => void;
}

export interface FilterData {
  websites: string[];
  from: Date;
  to: Date;
  categories: string[];
  authors: string[];
  title: string;
  content: string;
  sortBy: string;
}

export default function Filter({ onChange }: FilterProps) {
  const filterId = React.useId();
  const [data, setData] = useState<FilterData>({
    websites: [],
    from: new Date(0),
    to: new Date(),
    categories: [],
    authors: [],
    title: "",
    content: "",
    sortBy: "dateDesc",
  });

  const handleWebsiteChange = (websites: string[]) => {
    setData((prevState) => ({
      ...prevState,
      websites: [...websites],
    }));
  };

  const handleFromDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      from: event.target.value ? new Date(event.target.value) : new Date(0),
    }));
  };

  const handleToDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      to: event.target.value ? new Date(`${event.target.value}T23:59:59.999`) : new Date(),
    }));
  };

  const handleCategoriesChange = (categories: string[]) => {
    setData((prevState) => ({
      ...prevState,
      categories: [...categories],
    }));
  };

  const handleAuthorsChange = (authors: string[]) => {
    setData((prevState) => ({
      ...prevState,
      authors: [...authors],
    }));
  };

  const handleTitleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      title: event.target.value,
    }));
  };

  const handleContentChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      content: event.target.value,
    }));
  };

  const handleSortChange = (value: string) => {
    setData((prevState) => ({
      ...prevState,
      sortBy: value,
    }));
  };

  useEffect(() => {
    onChange(data);
  }, [data, onChange]);


  return (
    <div className="filters">
      <label htmlFor={`${filterId}-sort`}>Razvrsti po</label>
      <select id={`${filterId}-sort`} value={data.sortBy} onChange={event => handleSortChange(event.target.value)}>
        <option value="dateDesc">Novejše najprej</option><option value="dateAsc">Starejše najprej</option><option value="popularity">Popularnost</option><option value="views">Ogledi</option>
      </select>
      <fieldset className="filter-section"><legend>Viri novic</legend><div className="source-options">
        {[["rtvslo", "RTV Slovenija"], ["gov", "gov"], ["gov-vlada", "novice vlade"], ["24ur", "24 ur"], ["delo", "Delo"], ["mariborinfo", "Mariborinfo"], ["dnevnik", "Dnevnik"], ["svet", "Svet"], ["siol", "Siol"], ["sta", "STA"], ["n1", "N1"], ["ekipa24", "Ekipa 24"]].map(([value, label]) => (
          <label key={value}><input type="checkbox" value={value} defaultChecked onChange={event => {
            const inputs = event.currentTarget.closest("fieldset")!.querySelectorAll<HTMLInputElement>("input:checked");
            handleWebsiteChange(Array.from(inputs, input => input.value));
          }} />{label}</label>
        ))}
      </div></fieldset>
      <fieldset className="filter-section"><legend>Čas</legend><div className="date-fields"><label>Od datuma<input type="date" aria-label="Od datuma" onChange={handleFromDateChange} /></label><label>Do datuma<input type="date" aria-label="Do datuma" onChange={handleToDateChange} /></label></div></fieldset>
      <details className="filter-section" open><summary>Iskanje</summary>
        <label htmlFor={`${filterId}-title`}>Išči po naslovu</label><input id={`${filterId}-title`} type="search" placeholder="Naslov novice…" onChange={handleTitleChange} />
        <label htmlFor={`${filterId}-content`}>Išči po vsebini</label><input id={`${filterId}-content`} type="search" placeholder="Beseda v vsebini…" onChange={handleContentChange} />
      </details>
      <details className="filter-section"><summary>Kategorije in avtorji</summary>
        <label>Kategorije</label><DynamicInput onChange={handleCategoriesChange} />
        <label>Avtorji</label><DynamicInput onChange={handleAuthorsChange} />
      </details>
    </div>
  );
}

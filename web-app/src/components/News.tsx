import React, { useState, useEffect, useCallback } from "react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { getAll } from "../news/api";
import { FilterData } from "./Filter";
import FilterPanel from "./FilterPanel";
import * as FilterFn from "../news/filter";
import * as SortFn from "../news/sort";

export default function News() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [count, setCount] = useState<number>(50);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const presets = ["privzeto", "popularno", "vreme", "šport"];
  const [preset, setPreset] = useState<string>("privzeto");

  const presetsChange = (value: string) => {
    setPreset(value);
  };

  useEffect(() => {
    const fetchData = async () => {
      const data = await getAll();
      if (data) {
        setNews(data);
        setFilteredNews(data);
      }
      setFailed(!data);
      setLoading(false);
    };

    fetchData();
  }, []);

  useEffect(() => {
    const socket = io("ws://localhost:8000/news");
    socket.on("news-added", (newNews) => {
      setNews(newNews);
      setFilteredNews(newNews);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const handleFilterChange = useCallback((filterData: FilterData) => {
    let filtered = [...news];
    filterData.categories = filterData.categories.filter((item) => item !== "");
    filterData.authors = filterData.authors.filter((item) => item !== "");

    if(filterData.websites.length > 0) {
      filtered = FilterFn.websites(filtered, filterData.websites);
    }

    if (filterData.categories.length > 0) {
      filtered = FilterFn.categories(filtered, filterData.categories);
    }

    if (filterData.authors.length > 0) {
      filtered = FilterFn.authors(filtered, filterData.authors);
    }

    if (filterData.title.length > 0) {
      filtered = FilterFn.title(filtered, filterData.title);
    }

    if (filterData.content.length > 0) {
      filtered = FilterFn.content(filtered, filterData.content);
    }

    if (filterData.sortBy === "dateDesc") {
      filtered = SortFn.dateDesc(filtered);
    }

    if (filterData.sortBy === "dateAsc") {
      filtered = SortFn.dateAsc(filtered);
    }

    if (filterData.sortBy === "popularity" || preset === "popularno") {
      filtered = SortFn.popularity(filtered);
    }

    if (filterData.sortBy === "views") {
      filtered = SortFn.views(filtered);
    }

    filtered = FilterFn.date(filtered, filterData.from, filterData.to);
    filtered = FilterFn.categoryGroup(filtered, preset);
    setFilteredNews(filtered);
  }, [news, preset]);

  const handleCountChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newCount = parseInt(event.target.value, 10);
    setCount(newCount);
  };

  return (
    <>
      <h1 className="visually-hidden">Novice</h1>
      <div className="filter-layout">
        <FilterPanel onChange={handleFilterChange} />
        <section className="news-feed filter-content" aria-label="Novice">
          <div className="feed-toolbar">
            <div className="preset-tabs" role="radiogroup" aria-label="Prednastavitve">
              {presets.map((value) => <label key={value}><input className="visually-hidden" type="radio" name="customOptions" value={value} checked={preset === value} onChange={() => presetsChange(value)} /><span>{value === "privzeto" ? "Vse novice" : value === "popularno" ? "Popularno" : value === "vreme" ? "Vreme" : "Šport"}</span></label>)}
            </div>
          </div>
          <div className="results-summary" role="status">{loading ? "Nalaganje arhiva…" : failed ? "Arhiv ni dosegljiv" : `${filteredNews.length.toLocaleString("sl-SI")} novic v izboru`}<div className="count-control"><label htmlFor="news-count">Število novic</label><select id="news-count" value={count} onChange={handleCountChange}><option value={10}>10</option><option value={25}>25</option><option value={50}>50</option><option value={100}>100</option></select></div></div>
          {loading && <div className="loading-state" aria-label="Nalaganje novic"><span /><span /><span /></div>}
          {failed && <div className="empty-state" role="alert"><h2>Novic ni bilo mogoče naložiti.</h2><p>Preveri povezavo s strežnikom in poskusi znova.</p><button className="button" onClick={() => window.location.reload()}>Poskusi znova</button></div>}
          {!loading && !failed && filteredNews.length === 0 && <div className="empty-state"><h2>Ni novic za ta izbor.</h2><p>Spremeni filtre ali izberi drugo prednastavitev.</p></div>}
          {filteredNews.slice(0, count).map((article) => <NewsArticle key={article._id} article={article} />)}
        </section>
      </div>
    </>
  );
}

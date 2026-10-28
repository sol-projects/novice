import React, { useEffect, useState, useCallback } from "react";
import { io } from "socket.io-client";
import INews from "../news/model";
import { getAll } from "../news/api";
import * as Aggregate from "../news/aggregate";
import { Aggregation } from "../news/aggregate";
import { FilterData } from "./Filter";
import FilterPanel from "./FilterPanel";
import * as FilterFn from "../news/filter";
import { ResponsiveLine, Serie } from "@nivo/line";
import { ResponsivePie, MayHaveLabel } from "@nivo/pie";
import { ResponsiveWaffle, WaffleDatum } from "@nivo/waffle";
import { ResponsiveBar } from "@nivo/bar";
import { useAppearance } from "../appearance";

function dataToDisplay(chartType: string, news: INews[], n: number) {
  let aggregations: Aggregation[] = [];

  switch (chartType) {
    case "aggrCategories":
      aggregations = Aggregate.byTopCategories(news, n);
      break;
    case "aggrDates":
      aggregations = Aggregate.byDate(news, n);
      break;
    case "aggrAuthors":
      aggregations = Aggregate.byTopAuthors(news, n);
      break;
    case "aggrMonths":
      aggregations = Aggregate.byMonths(news, n);
      break;
    default:
      aggregations = Aggregate.byTopCategories(news, n);
      break;
  }

  return aggregations;
}

function getDataForLineChart(aggregations: Aggregation[]): Serie[] {
  return [
    {
      id: "Število novic",
      data: aggregations.map((aggregation) => ({
        x: aggregation.key,
        y: aggregation.value,
      })),
    },
  ];
}

function getDataForPieChart(aggregations: Aggregation[]): MayHaveLabel[] {
  return aggregations.map((aggregation) => ({
    id: aggregation.key,
    value: aggregation.value,
  })) as MayHaveLabel[];
}

function getDataForWaffleChart(aggregations: Aggregation[]): WaffleDatum[] {

  return aggregations.map((aggregation) => ({
    id: aggregation.key,
    value: aggregation.value,
    label: `${aggregation.key}: ${aggregation.value}`,
  }));
}


export default function Chart() {
  useAppearance();
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [value, setValue] = useState("aggrCategories");
  const [nResults, setNResults] = useState(10);
  const [chartType, setChartType] = useState("LineChart");
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const get = async () => {
      const data = await getAll();
      if (data) setNews(data);
      setFailed(!data);
      setLoading(false);
    };
    get();
    const socket = io("ws://localhost:8000/news");
    socket.on("news-added", (newNews) => setNews(newNews));
    return () => { socket.disconnect(); };
  }, []);

  const handleChartOptionChange = (value: string) => setValue(value);
  const handleFilterChange = useCallback((filterData: FilterData) => {
    let filtered = news;
    const categories = filterData.categories.filter(Boolean);
    const authors = filterData.authors.filter(Boolean);
    if (filterData.websites.length > 0) filtered = FilterFn.websites(filtered, filterData.websites);
    if (categories.length > 0) filtered = FilterFn.categories(filtered, categories);
    if (authors.length > 0) filtered = FilterFn.authors(filtered, authors);
    if (filterData.title.length > 0) filtered = FilterFn.title(filtered, filterData.title);
    if (filterData.content.length > 0) filtered = FilterFn.content(filtered, filterData.content);
    filtered = FilterFn.date(filtered, filterData.from, filterData.to);
    setFilteredNews(filtered);
  }, [news]);

  const aggregations = dataToDisplay(value, filteredNews, nResults);
  const totalValue = aggregations.reduce((sum, item) => sum + item.value, 0);
  const labels: Record<string, string> = { aggrCategories: "Kategorije", aggrDates: "Datum", aggrAuthors: "Avtorji", aggrMonths: "Meseci" };
  const colors = [1, 2, 3, 4].map(index => getComputedStyle(document.documentElement).getPropertyValue(`--color-chart-${index}`).trim());
  const chartTheme = { textColor: "var(--color-ink)", fontSize: 12, fontFamily: "var(--font-display)", grid: { line: { stroke: "var(--color-rule)" } }, tooltip: { container: { background: "var(--color-sheet)", color: "var(--color-ink)", borderRadius: 0 } } };
  const axisBottom = { legend: labels[value], legendOffset: 52, legendPosition: "middle" as const, tickRotation: -25, format: (v: string | number) => String(v).length > 13 ? `${String(v).slice(0, 13)}…` : String(v) };
  const axisLeft = { legend: "Število novic", legendOffset: -44, legendPosition: "middle" as const };
  const margin = { top: 24, right: 28, bottom: 100, left: 60 };

  return (
    <>
      <h1 className="visually-hidden">Grafi novic</h1>
      <div className="filter-layout">
        <FilterPanel onChange={handleFilterChange} />
        <section className="chart-workspace filter-content" aria-label="Analiza novic">
          <div className="chart-controls">
            <div><label htmlFor="nResultsInput">Število podatkov</label><input type="number" id="nResultsInput" min={1} max={100} value={nResults} onChange={event => setNResults(Math.max(1, Math.min(100, Number(event.target.value) || 1)))} /></div>
            <div><label htmlFor="chart-type">Vrsta grafa</label><select id="chart-type" value={chartType} onChange={event => setChartType(event.target.value)}><option value="LineChart">Črtni</option><option value="BarChart">Stolpčni</option><option value="PieChart">Tortni</option><option value="WaffleChart">Mrežni</option></select></div>
          </div>
          <fieldset className="chart-dimensions"><legend>Združi po</legend>{Object.entries(labels).map(([key, label]) => <label key={key}><input type="radio" name="aggregation" value={key} checked={value === key} onChange={() => handleChartOptionChange(key)} /><span>{label}</span></label>)}</fieldset>
          <div className="workspace-label"><h2>{labels[value]}</h2><span role="status">{loading ? "Nalaganje…" : `${filteredNews.length} novic v izboru`}</span></div>
          <div className="chart-plot" aria-label={`Graf: ${labels[value]}`}>
            {loading ? <div className="loading-state" aria-label="Nalaganje grafa"><span /><span /><span /></div> : failed ? <div className="empty-state" role="alert"><h2>Podatki niso dosegljivi.</h2><p>Preveri povezavo s strežnikom.</p><button className="button" onClick={() => window.location.reload()}>Poskusi znova</button></div> : !aggregations.length ? <div className="empty-state"><h2>Ni podatkov za ta izbor.</h2><p>Spremeni filtre za prikaz grafa.</p></div> : <>
              {chartType === "LineChart" && <ResponsiveLine data={getDataForLineChart(aggregations)} margin={margin} theme={chartTheme} colors={colors} xScale={{ type: "point" }} yScale={{ type: "linear", min: 0, max: "auto" }} axisBottom={axisBottom} axisLeft={axisLeft} enableGridX={false} pointSize={7} useMesh enableSlices="x" animate={false} role="img" />}
              {chartType === "BarChart" && <ResponsiveBar data={aggregations} keys={["value"]} indexBy="key" margin={margin} theme={chartTheme} colors={colors} axisBottom={axisBottom} axisLeft={axisLeft} padding={0.4} enableLabel={false} animate={false} role="img" />}
              {chartType === "PieChart" && <ResponsivePie data={getDataForPieChart(aggregations)} margin={{ top: 40, right: 70, bottom: 40, left: 70 }} theme={chartTheme} colors={colors} innerRadius={0.6} padAngle={1} arcLabelsTextColor="var(--color-accent-ink)" arcLabelsSkipAngle={10} animate={false} role="img" />}
              {chartType === "WaffleChart" && <ResponsiveWaffle data={getDataForWaffleChart(aggregations)} total={totalValue} rows={15} columns={15} margin={margin} colors={colors} role="img" />}
            </>}
          </div>
          <p className="chart-caption">Število novic po izbranem kriteriju. Vir: trenutni izbor arhiva.</p>
        </section>
      </div>
    </>
  );
}

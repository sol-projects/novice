import React, { useEffect, useState } from "react";
import {
  VStack,
  Center,
  RadioGroup,
  Radio,
  Heading,
  Input,
  InputGroup,
  InputLeftAddon,
} from "@chakra-ui/react";
import { io } from "socket.io-client";
import INews from "../news/model";
import { getAll } from "../news/api";
import * as Aggregate from "../news/aggregate";
import { ResponsiveLine } from '@nivo/line'
import { Aggregation } from "../news/aggregate";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";

type ChartInfo = {
  bottom: string;
  left: string;
  dataAmount: number;
};

let chartInfo: ChartInfo = {
  bottom: "categories",
  left: "number of articles",
  dataAmount: 10,
};

function dataToDisplay(option: string, news: INews[], n: number) {
  var aggregations: Aggregation[] = [];

  switch (option) {
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

  return [
    {
      id: chartInfo.left,
      data: aggregations.map((aggregation) => ({
        x: aggregation.key,
        y: aggregation.value,
      })),
    },
  ];
}

export default function Chart() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [value, setValue] = useState("aggrCategories");
  const [nResults, setNResults] = useState(10);

  useEffect(() => {
    const get = async () => {
      const data = await getAll();
      if (data) {
        setNews(data);
      }
    };

    get();
  }, []);

  useEffect(() => {
    const socket = io("ws://localhost:8000/news");
    socket.on("news-added", (newNews) => {
      setNews(newNews);
    });
    return () => {
      socket.disconnect();
    };
  }, []);

  const handleChartOptionChange = (value: string) => {
    setValue(value);
  };

  const handleFilterChange = (filterData: FilterData) => {
    let filtered = news;
    filterData.categories = filterData.categories.filter((item) => item !== "");
    filterData.authors = filterData.authors.filter((item) => item !== "");
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

    setFilteredNews(filtered);
  };

  return (
    <Center>
      <VStack width="80%" height="600px">
        <Filter onChange={handleFilterChange} />
        <RadioGroup onChange={handleChartOptionChange} value={value}>
          <Radio value="aggrCategories">kategorije</Radio>
          <Radio value="aggrDates">datum</Radio>
          <Radio value="aggrAuthors">avtorji</Radio>
          <Radio value="aggrMonths">meseci</Radio>
        </RadioGroup>
        <InputGroup width="35%">
          <InputLeftAddon children="število podatkov:" />
          <Input
            type="number"
            placeholder="number"
            value={nResults}
            onChange={(event) => setNResults(Number(event.target.value))}
          />
        </InputGroup>
        <ResponsiveLine
          data={dataToDisplay(value, filteredNews, nResults)}
          margin={{ top: 50, right: 200, bottom: 100, left: 100 }}
          xScale={{ type: 'point' }}
          yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false, reverse: false }}
          axisTop={null}
          axisRight={null}
          axisBottom={{
            legend: chartInfo.bottom,
            legendOffset: 36,
            legendPosition: 'middle',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
          }}
          axisLeft={{
            legend: chartInfo.left,
            legendOffset: -40,
            legendPosition: 'middle',
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
          }}
          colors={{ scheme: 'nivo' }}
          pointSize={10}
          pointColor={{ theme: 'background' }}
          pointBorderWidth={2}
          pointBorderColor={{ from: 'serieColor' }}
          pointLabel="y"
          pointLabelYOffset={-12}
          useMesh={true}
        />

      </VStack>
    </Center>
  );
}

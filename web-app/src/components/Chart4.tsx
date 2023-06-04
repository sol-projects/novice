import React, { useEffect, useState } from "react";
import {
  VStack,
  Center,
  RadioGroup,
  Radio,
  Stack,
  Heading,
  InputLeftAddon,
  Input,
  InputGroup,
} from "@chakra-ui/react";
import { io } from "socket.io-client";
import { ResponsiveWaffle } from "@nivo/waffle";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import * as Aggregate from "../news/aggregate";
import { Aggregation } from "../news/aggregate";
import INews from "../news/model";

type ChartInfo = {
  bottom: string;
  left: string;
  dataAmount: number;
};

type WaffleChartData = {
  id: string;
  value: number;
};

const chartInfo: ChartInfo = {
  bottom: "categories",
  left: "number of articles",
  dataAmount: 10,
};

const dataToDisplay = (option: string, news: INews[], n: number): WaffleChartData[] => {
  let aggregations: Aggregation[] = [];

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

  return aggregations.map((aggregation) => ({
    id: aggregation.key,
    value: aggregation.value,
    label: `${aggregation.key}: ${aggregation.value}`, // Add label property
  }));
};


const Chart = () => {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [value, setValue] = useState<string>("aggrCategories");
  const [nResults, setNResults] = useState<number>(10);

  useEffect(() => {
    const fetchData = async () => {
      const data = await getAll();
      if (data) {
        setNews(data);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    const socket = io("ws://localhost:8000/news");
    socket.on("news-added", (newNews: INews[]) => {
      setNews(newNews);
    });
    return () => {
      socket.disconnect();
    };
  }, []);

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

  const waffleChartData = dataToDisplay(value, filteredNews, nResults);
  const totalValue = waffleChartData.reduce((sum, data) => sum + data.value, 0);

  return (
    <Center>
      <VStack width="80%" height="600px">
        <Filter onChange={handleFilterChange} />
        <RadioGroup onChange={setValue} value={value}>
          <Stack direction="row">
            <Radio value="aggrCategories">Categories</Radio>
            <Radio value="aggrDates">Dates</Radio>
            <Radio value="aggrAuthors">Authors</Radio>
            <Radio value="aggrMonths">Months</Radio>
            <InputGroup width="35%">
              <InputLeftAddon children="Number of Data:" />
              <Input
                type="number"
                id="nResultsInput"
                placeholder="Number"
                value={nResults}
                onChange={(event) => setNResults(Number(event.target.value))}
              />
            </InputGroup>
          </Stack>
        </RadioGroup>
        
        <ResponsiveWaffle
          data={waffleChartData}
          total={totalValue}
          rows={10}
          columns={10}
          margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
          colors={{ scheme: "category10" }}
          borderColor={{ theme: "background" }}
          legends={[
            {
              anchor: "bottom",
              direction: "row",
              justify: false,
              translateX: 0,
              translateY: 80,
              itemsSpacing: 2,
              itemWidth: 120,
              itemHeight: 20,
              itemDirection: "left-to-right",
              itemOpacity: 0.8,
              symbolSize: 20,
              symbolShape: "circle",
              effects: [
                {
                  on: "hover",
                  style: {
                    itemOpacity: 1,
                  },
                },
              ],
              items: waffleChartData.map((data) => ({
                id: data.id,
                value: data.value,
              })),
            },
          ]}
        />
      </VStack>
    </Center>
  );
};

export default Chart;

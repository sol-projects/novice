import React from "react";
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
import { useEffect, useState } from "react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { ResponsivePie } from "@nivo/pie";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import * as Aggregate from "../news/aggregate";
import { Aggregation } from "../news/aggregate";

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
type MayHaveLabel = {
  id: string;
  label?: string;
  value: number;
};


function dataToDisplay(option: string, news: INews[], n: number): MayHaveLabel[] {
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
  }));
}



export default function Chart() {
  const [news, set] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [value, setValue] = useState("aggrCategories");
  const [nResults, setNResults] = useState(10);

  useEffect(() => {
    const get = async () => {
      const data = await getAll();
      if (data) {
        set(data);
      }
    };

    get();
  }, []);

  useEffect(() => {
    const socket = io("ws://localhost:8000/news");
    socket.on("news-added", (newNews) => {
      set(newNews);
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

  return (
    <Center>
      <VStack width="80%" height="600px">
        <Filter onChange={handleFilterChange} />
        <RadioGroup onChange={setValue} value={value}>
          <Stack direction="row">
            <Radio value="aggrCategories">kategorije</Radio>
            <Radio value="aggrDates">datum</Radio>
            <Radio value="aggrAuthors">avtorji</Radio>
            <Radio value="aggrMonths">meseci</Radio>
            <InputGroup width="35%">
              <InputLeftAddon children="število podatkov:" />
              <Input
                type="number"
                id="nResultsInput" // Unique id for the input field
                placeholder="number"
                value={nResults}
                onChange={(event) => setNResults(Number(event.target.value))}
              />

            </InputGroup>
          </Stack>
        </RadioGroup>
        <ResponsivePie
          data={dataToDisplay(value, filteredNews, nResults)}
          margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
          innerRadius={0.5}
          padAngle={0.7}
          cornerRadius={3}
          colors={{ scheme: "category10" }}
          enableArcLabels={true}
          arcLabelsSkipAngle={10}
          arcLabelsTextColor="#333333"
        />




      </VStack>
    </Center>
  );
}

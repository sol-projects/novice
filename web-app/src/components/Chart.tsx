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
import { ResponsiveBar } from "@nivo/bar";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import * as Aggregate from "../news/aggregate";

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
  switch (option) {
    case "aggrCategories":
      chartInfo.bottom = "kategorija";
      chartInfo.left = "število člankov";
      return Aggregate.byTopCategories(news, n);
    case "aggrDates":
      chartInfo.bottom = "datum";
      chartInfo.left = "število člankov";
      return Aggregate.byDate(news, n);
    case "aggrMonths":
      chartInfo.bottom = "mesec";
      chartInfo.left = "število člankov";
      return Aggregate.byMonths(news, n);
    case "aggrAuthors":
      chartInfo.bottom = "avtor";
      chartInfo.left = "število člankov";
      return Aggregate.byTopAuthors(news, n);
    default:
      chartInfo.bottom = "avtor";
      chartInfo.left = "število člankov";
      return Aggregate.byTopCategories(news, n);
  }
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
                placeholder="number"
                value={nResults}
                onChange={(event) => setNResults(Number(event.target.value))}
              />
            </InputGroup>
          </Stack>
        </RadioGroup>
        <ResponsiveBar
          data={dataToDisplay(value, filteredNews, nResults)}
          keys={["value"]}
          indexBy="key"
          margin={{ top: 50, right: 200, bottom: 100, left: 100 }}
          padding={0.3}
          valueScale={{ type: "linear" }}
          indexScale={{ type: "band", round: true }}
          colors={{ scheme: "nivo" }}
          defs={[
            {
              id: "dots",
              type: "patternDots",
              background: "inherit",
              color: "#38bcb2",
              size: 4,
              padding: 1,
              stagger: true,
            },
            {
              id: "lines",
              type: "patternLines",
              background: "inherit",
              color: "#eed312",
              rotation: -45,
              lineWidth: 6,
              spacing: 10,
            },
          ]}
          borderColor={{
            from: "color",
            modifiers: [["darker", 1.6]],
          }}
          axisTop={null}
          axisRight={null}
          axisBottom={{
            format: (v) => {
              return v.length > 10 ? `${v.substring(0, 10)}...` : v;
            },
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: chartInfo.bottom,
            legendPosition: "middle",
            legendOffset: 32,
          }}
          axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: chartInfo.left,
            legendPosition: "middle",
            legendOffset: -40,
          }}
          labelSkipWidth={12}
          labelSkipHeight={12}
          labelTextColor={{
            from: "color",
            modifiers: [["darker", 1.6]],
          }}
          /*legends={[
            {
                dataFrom: 'indexes',
                anchor: 'bottom-right',
                direction: 'column',
                justify: false,
                translateX: 120,
                translateY: 0,
                itemsSpacing: 2,
                itemWidth: 100,
                itemHeight: 20,
                itemDirection: 'left-to-right',
                itemOpacity: 0.85,
                symbolSize: 20,
                effects: [
                    {
                        on: 'hover',
                        style: {
                            itemOpacity: 1
                        }
                    }
                ]
            }
        ]}*/
          role="application"
          //ariaLabel="Nivo bar chart demo"
          //barAriaLabel={e=>e.id+": "+e.formattedValue+" in country: "+e.indexValue}
        />
      </VStack>
    </Center>
  );
}

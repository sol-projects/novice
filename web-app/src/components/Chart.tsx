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
  Select,
} from "@chakra-ui/react";
import { io } from "socket.io-client";
import INews from "../news/model";
import { getAll } from "../news/api";
import * as Aggregate from "../news/aggregate";
import { ResponsiveLine } from "@nivo/line";
import { Aggregation } from "../news/aggregate";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import { ResponsivePie } from "@nivo/pie";
import { ResponsiveWaffle } from "@nivo/waffle";
import { Serie } from "@nivo/line";
import { MayHaveLabel } from "@nivo/pie";
import { WaffleDatum } from "@nivo/waffle";
import { ResponsiveBar } from "@nivo/bar";

type ChartInfo = {
  bottom: string;
  left: string;
  dataAmount: number;
};

let chartInfo: ChartInfo = {
  bottom: "Kategorije",
  left: "Število novic",
  dataAmount: 10,
};


type BarData = {
  [key: string]: string | number;
};


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

function getDataForBarChart(aggregations: Aggregation[]): BarData[] {
  return aggregations.map((aggregation) => ({
    [chartInfo.bottom]: aggregation.key,
    [chartInfo.left]: aggregation.value,
  }));
}


function getDataForLineChart(aggregations: Aggregation[]): Serie[] {
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

function getDataForPieChart(aggregations: Aggregation[]): MayHaveLabel[] {
  return aggregations.map((aggregation) => ({
    id: aggregation.key,
    value: aggregation.value,
  })) as MayHaveLabel[];
}


function getDataForWaffleChart(aggregations: Aggregation[]): WaffleDatum[] {
  const colors = generateColors(aggregations.length); // Generate colors based on the number of aggregations

  return aggregations.map((aggregation) => ({
    id: aggregation.key,
    value: aggregation.value,
    label: `${aggregation.key}: ${aggregation.value}`, // Add label property
  }));
}


function generateColors(count: number): string[] {
  // Generate colors dynamically based on the count
  const colors = [];
  for (let i = 0; i < count; i++) {
    // Generate random color codes or use a predefined color scheme
    // Example: Generate random colors using hexadecimal format
    const color = "#" + Math.floor(Math.random() * 16777215).toString(16);
    colors.push(color);
  }
  return colors;
}




export default function Chart() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [value, setValue] = useState("aggrCategories");
  const [nResults, setNResults] = useState(10);
  const [chartType, setChartType] = useState("LineChart");
  const [totalValue, setTotalValue] = useState(0);

  useEffect(() => {
    const get = async () => {
      const data = await getAll();
      if (data) {
        setNews(data);
      }
    };

    get();
  }, [filteredNews, nResults, value]);

  useEffect(() => {
    setTotalValue(
      dataToDisplay(value, filteredNews, nResults).reduce(
        (sum, aggregation) => sum + aggregation.value,
        0
      )
    );
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

  let chartData: Serie[] | MayHaveLabel[] | WaffleDatum[] = [];

if (chartType === "LineChart") {
  chartData = getDataForLineChart(dataToDisplay(value, filteredNews, nResults));
} else if (chartType === "BarChart") {
  chartData = getDataForBarChart(dataToDisplay(value, filteredNews, nResults));
} else if (chartType === "PieChart") {
  chartData = getDataForPieChart(dataToDisplay(value, filteredNews, nResults));
} else if (chartType === "WaffleChart") {
  const waffleChartData = dataToDisplay(value, filteredNews, nResults);
  chartData = getDataForWaffleChart(waffleChartData) as WaffleDatum[];
}

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
          <InputLeftAddon children="število podatkov: " />
          <Input
            type="number"
            id="nResultsInput"
            placeholder="number"
            value={nResults}
            onChange={(event) => setNResults(Number(event.target.value))}
          />
        </InputGroup>
        <Select
          onChange={(e) => setChartType(e.target.value)}
          value={chartType}
        >
          <option value="LineChart">Line chart</option>
          <option value="BarChart">Bar chart</option>
          <option value="PieChart">Pie chart</option>
        </Select>

        {chartType === "LineChart" && (
          <ResponsiveLine
            data={chartData as Serie[]}
            margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
            xScale={{ type: "point" }}
            yScale={{
              type: "linear",
              min: "auto",
              max: "auto",
              stacked: true,
              reverse: false,
            }}
            yFormat=" >-.2f"
            axisTop={null}
            axisRight={null}
            axisBottom={{
              legend: chartInfo.bottom,
              legendOffset: 36,
              legendPosition: "middle",
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
            }}
            axisLeft={{
              legend: chartInfo.left,
              legendOffset: -40,
              legendPosition: "middle",
              tickSize: 5,
              tickPadding: 5,
              tickRotation: 0,
            }}
            colors={{ scheme: "category10" }}
            pointSize={10}
            pointColor={{ theme: "background" }}
            pointBorderWidth={2}
            pointBorderColor={{ from: "serieColor" }}
            pointLabelYOffset={-12}
            useMesh={true}
            enableSlices="x"
            enableGridX={false}
            enableGridY={true}
          />
        )}
        {chartType === "BarChart" && (
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
        )}
        {chartType === "PieChart" && (
          <ResponsivePie
          data={chartData as MayHaveLabel[]}
          margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
          innerRadius={0.5}
          padAngle={0.7}
          cornerRadius={3}
          colors={{ scheme: "category10" }}
          enableArcLabels={true}
          arcLabelsSkipAngle={10}
          arcLabelsTextColor="#333333"
        />
        )}
        {chartType === "WaffleChart" && (
         <ResponsiveWaffle
         data={chartData as WaffleDatum[]}
         total={totalValue}
         rows={15}
         columns={15}
         margin={{ top: 40, right: 80, bottom: 80, left: 80 }}
         colors={{ scheme: "category10" }}
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
             items: (chartData as WaffleDatum[]).map((data, index) => ({
               id: `${index}`,
               value: data.value,
             })),
           },
         ]}
       />
        )}
      </VStack>
    </Center>
  );
}

import React from "react";
import { VStack, Center } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { ResponsiveLine } from "@nivo/line";
import { getAll } from "../news/api";

export default function Chart() {
  const [news, set] = useState<INews[]>([]);
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
  }, []);

  return (
    <ResponsiveLine
      data={[]}
      margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
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
        tickSize: 5,
        tickPadding: 5,
        tickRotation: 0,
        legend: "transportation",
        legendOffset: 36,
        legendPosition: "middle",
      }}
      axisLeft={{
        tickSize: 5,
        tickPadding: 5,
        tickRotation: 0,
        legend: "count",
        legendOffset: -40,
        legendPosition: "middle",
      }}
      pointSize={10}
      pointColor={{ theme: "background" }}
      pointBorderWidth={2}
      pointBorderColor={{ from: "serieColor" }}
      pointLabelYOffset={-12}
      useMesh={true}
      legends={[
        {
          anchor: "bottom-right",
          direction: "column",
          justify: false,
          translateX: 100,
          translateY: 0,
          itemsSpacing: 0,
          itemDirection: "left-to-right",
          itemWidth: 80,
          itemHeight: 20,
          itemOpacity: 0.75,
          symbolSize: 12,
          symbolShape: "circle",
          symbolBorderColor: "rgba(0, 0, 0, .5)",
          effects: [
            {
              on: "hover",
              style: {
                itemBackground: "rgba(0, 0, 0, .03)",
                itemOpacity: 1,
              },
            },
          ],
        },
      ]}
    />
  );
}

import React from "react";
import { VStack, Center } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";

export default function News() {
  const [news, set] = useState<any[]>([]);

  useEffect(() => {
    const get = async () => {
      try {
        const response = await fetch("http://localhost:8000/news");
        if (!response.ok) {
          console.log("Failed to fetch orders");
        }

        const data = await response.json();
        data.sort(function (a: any, b: any) {
          const c = new Date(a.date);
          const d = new Date(b.date);
          return d.getTime() - c.getTime();
        });
        set(data);
      } catch (error) {
        console.error("Error fetching news:", error);
      }
    };

    get();
  }, []);

  useEffect(() => {
    const socket = io("ws://localhost:8000/news");

    socket.on("connnection", () => {
      console.log("connected to server");
    });

    socket.on("news-added", (newNews) => {
      set(newNews);
    });

    socket.on("message", (message) => {
      console.log(message);
    });

    socket.on("disconnect", () => {
      console.log("Socket disconnecting");
    });
  }, []);

  return (
    <Center>
      <VStack width="80%">
        {news && news.map((article) => <NewsArticle article={article} />)}
      </VStack>
    </Center>
  );
}

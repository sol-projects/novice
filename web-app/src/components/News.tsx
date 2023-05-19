import React from "react";
import { VStack, Center, HStack } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter from "./Filter";

export default function News() {
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
    <>
      <Center>
        <VStack width="80%">
          <Filter />
          {news && news.map((article) => <NewsArticle article={article} />)}
        </VStack>
      </Center>
    </>
  );
}

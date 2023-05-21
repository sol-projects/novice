import React, { useState, useEffect } from "react";
import { VStack, Center } from "@chakra-ui/react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";

export default function News() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      const data = await getAll();
      if (data) {
        setNews(data);
        setFilteredNews(data);
      }
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

    setFilteredNews(filtered);
  };

  return (
    <>
      <Center>
        <VStack width="80%">
          <Filter onChange={handleFilterChange} />
          {filteredNews.map((article) => (
            <NewsArticle key={article._id} article={article} />
          ))}
        </VStack>
      </Center>
    </>
  );
}

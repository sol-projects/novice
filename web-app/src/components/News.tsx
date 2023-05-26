import React, { useState, useEffect } from "react";
import { VStack, Center, Select } from "@chakra-ui/react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import * as SortFn from "../news/sort";

export default function News() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [count, setCount] = useState<number>(50);

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
    let filtered = [...news];
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

    if (filterData.sortBy === "dateDesc") {
      filtered = SortFn.dateDesc(filtered);
    }

    if (filterData.sortBy === "dateAsc") {
      filtered = SortFn.dateAsc(filtered);
    }

    setFilteredNews(filtered);
  };

  const handleCountChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newCount = parseInt(event.target.value, 10);
    setCount(newCount);
  };

  return (
    <>
      <Center>
        <VStack width="80%">
          <VStack width="20%">
            <Filter onChange={handleFilterChange} />
            <Select
              width="auto"
              minWidth="min-content"
              value={count}
              onChange={handleCountChange}
            >
              <option value={5}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </Select>
          </VStack>
          {filteredNews.slice(0, count).map((article) => (
            <NewsArticle key={article._id} article={article} />
          ))}
        </VStack>
      </Center>
    </>
  );
}

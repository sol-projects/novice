import React, { useState, useEffect } from "react";
import {
  VStack,
  Center,
  Select,
  HStack,
  useRadioGroup,
} from "@chakra-ui/react";
import { io } from "socket.io-client";
import NewsArticle from "./NewsArticle";
import INews from "../news/model";
import { getAll } from "../news/api";
import Filter, { FilterData } from "./Filter";
import * as FilterFn from "../news/filter";
import * as SortFn from "../news/sort";
import RadioCard from "./RadioCard";

export default function News() {
  const [news, setNews] = useState<INews[]>([]);
  const [filteredNews, setFilteredNews] = useState<INews[]>([]);
  const [count, setCount] = useState<number>(50);
  const presets = ["privzeto", "popularno", "vreme", "šport"];
  const [preset, setPreset] = useState<string>("privzeto");

  const presetsChange = (value: string) => {
    setPreset(value);
  };

  const { getRootProps, getRadioProps } = useRadioGroup({
    name: "customOptions",
    defaultValue: "popularno",
    onChange: presetsChange,
  });

  const group = getRootProps();

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

    if(filterData.websites.length > 0) {
      filtered = FilterFn.websites(filtered, filterData.websites);
    }

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

    if (filterData.sortBy === "popularity" || preset === "popularno") {
      filtered = SortFn.popularity(filtered);
    }

    if (filterData.sortBy === "views") {
      filtered = SortFn.views(filtered);
    }

    filtered = FilterFn.date(filtered, filterData.from, filterData.to);
    filtered = FilterFn.categoryGroup(filtered, preset);
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
          <HStack width="20%">
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
          </HStack>

          <HStack {...group}>
            {presets.map((value) => {
              const radio = getRadioProps({ value });
              return (
                <RadioCard key={value} {...radio}>
                  {value}
                </RadioCard>
              );
            })}
          </HStack>
          {filteredNews.slice(0, count).map((article) => (
            <NewsArticle key={article._id} article={article} />
          ))}
        </VStack>
      </Center>
    </>
  );
}

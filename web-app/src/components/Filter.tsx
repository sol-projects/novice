import React from "react";
import {
  Button,
  Input,
  FormControl,
  FormLabel,
  HStack,
  VStack,
  Heading,
  useDisclosure,
  Checkbox,
  CheckboxGroup,
  AccordionPanel,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionIcon,
  Radio,
  RadioGroup,
  Stack,
  Box,
} from "@chakra-ui/react";
import { useState, useEffect, useRef } from "react";
import DynamicInput from "./DynamicInput";

export interface FilterProps {
  onChange: (filterData: FilterData) => void;
}

export interface FilterData {
  websites: string[];
  from: Date;
  to: Date;
  categories: string[];
  authors: string[];
  title: string;
  content: string;
  sortBy: string;
}

export default function Filter({ onChange }: FilterProps) {
  const [data, setData] = useState<FilterData>({
    websites: [],
    from: new Date(),
    to: new Date(),
    categories: [],
    authors: [],
    title: "",
    content: "",
    sortBy: "dateDesc",
  });

  const handleWebsiteChange = (websites: string[]) => {
    setData((prevState) => ({
      ...prevState,
      websites: [...websites],
    }));
  };

  const handleFromDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      from: new Date(event.target.value),
    }));
  };

  const handleToDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      to: new Date(event.target.value),
    }));
  };

  const handleCategoriesChange = (categories: string[]) => {
    setData((prevState) => ({
      ...prevState,
      categories: [...categories],
    }));
  };

  const handleAuthorsChange = (authors: string[]) => {
    setData((prevState) => ({
      ...prevState,
      authors: [...authors],
    }));
  };

  const handleTitleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      title: event.target.value,
    }));
  };

  const handleContentChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setData((prevState) => ({
      ...prevState,
      content: event.target.value,
    }));
  };

  const handleSortChange = (value: string) => {
    setData((prevState) => ({
      ...prevState,
      sortBy: value,
    }));
  };

  useEffect(() => {
    onChange(data);
  }, [data, onChange]);

  return (
    <Accordion allowToggle>
      <AccordionItem>
        <h2>
          <AccordionButton>
            <Box as="span" flex="1" textAlign="left">
              filtriranje
            </Box>
            <AccordionIcon />
          </AccordionButton>
        </h2>
        <AccordionPanel pb={4}>
          <FormControl>
            <RadioGroup
              colorScheme="green"
              onChange={handleSortChange}
              value={data.sortBy}
            >
              <Stack direction="row">
                <FormLabel>Sortiraj po: </FormLabel>
                <Radio value="dateDesc">Novejše</Radio>
                <Radio value="dateAsc">Starejše</Radio>
              </Stack>
            </RadioGroup>
            <CheckboxGroup
              colorScheme="green"
              defaultValue={[
                "gov",
                "rtvslo",
                "gov-vlada",
                "24ur",
                "delo",
                "mariborinfo",
                "svet",
                "sta",
                "siol",
                "ekipa24",
                "dnevnik",
                "n1",
              ]}
              onChange={handleWebsiteChange}
            >
              <VStack align="stretch">
                <HStack>
                  <Checkbox width="50%" value="rtvslo">
                    RTV Slovenija
                  </Checkbox>
                  <Checkbox value="gov">gov</Checkbox>
                </HStack>
                <HStack>
                  <Checkbox width="50%" value="gov-vlada">
                    novice vlade
                  </Checkbox>
                  <Checkbox value="24ur">24 ur</Checkbox>
                </HStack>
                <HStack>
                  <Checkbox width="50%" value="delo">
                    Delo
                  </Checkbox>
                  <Checkbox value="mariborinfo">Mariborinfo</Checkbox>
                </HStack>
                <HStack>
                  <Checkbox width="50%" value="dnevnik">
                    Dnevnik
                  </Checkbox>
                  <Checkbox value="svet">Svet</Checkbox>
                </HStack>
                <HStack>
                  <Checkbox width="50%" value="siol">
                    Siol
                  </Checkbox>
                  <Checkbox value="sta">STA</Checkbox>
                </HStack>
                <HStack>
                  <Checkbox width="50%" value="n1">
                    N1
                  </Checkbox>
                  <Checkbox value="ekipa24">Ekipa 24</Checkbox>
                </HStack>
              </VStack>
            </CheckboxGroup>
            <FormLabel>Čas</FormLabel>
            <HStack>
              <Input type="date" onChange={handleFromDateChange} />
              <Input type="date" onChange={handleToDateChange} />
            </HStack>
            <FormLabel>Kategorije</FormLabel>
            <DynamicInput onChange={handleCategoriesChange} />
            <FormLabel>Avtorji</FormLabel>
            <DynamicInput onChange={handleAuthorsChange} />

            <FormLabel>Išči po naslovu</FormLabel>
            <Input type="text" onChange={handleTitleChange} />
            <FormLabel>Išči po vsebini</FormLabel>
            <Input type="text" onChange={handleContentChange} />
          </FormControl>
        </AccordionPanel>
      </AccordionItem>
    </Accordion>
  );
}

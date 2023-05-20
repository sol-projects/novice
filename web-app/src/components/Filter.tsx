import React from "react";
import {
  Drawer,
  DrawerBody,
  DrawerFooter,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
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
}

export default function Filter({ onChange }: FilterProps) {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const btnRef = React.useRef<HTMLButtonElement>(null);
  const [data, setData] = useState<FilterData>({
    websites: [],
    from: new Date(),
    to: new Date(),
    categories: [],
    authors: [],
    title: "",
    content: "",
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

  useEffect(() => {
    onChange(data);
  }, [data, onChange]);

  return (
    <>
      <Button ref={btnRef} colorScheme="teal" onClick={onOpen}>
        Sort & Filter
      </Button>
      <Drawer
        isOpen={isOpen}
        placement="left"
        onClose={onClose}
        finalFocusRef={btnRef}
      >
        <DrawerOverlay />
        <DrawerContent>
          <DrawerCloseButton />
          <DrawerHeader>Filter</DrawerHeader>

          <DrawerBody>
            <FormControl>
              <FormLabel>Website</FormLabel>
              <CheckboxGroup
                colorScheme="green"
                defaultValue={[
                  "gov",
                  "rtvslo",
                  "gov-vlada",
                  "24ur",
                  "delo",
                  "mbinfo",
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
                    <Checkbox value="mbinfo">Mariborinfo</Checkbox>
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
              <FormLabel>Date</FormLabel>
              <HStack>
                <Input type="date" onChange={handleFromDateChange} />
                <Input type="date" onChange={handleToDateChange} />
              </HStack>
              <FormLabel>Categories</FormLabel>
              <DynamicInput onChange={handleCategoriesChange} />
              <FormLabel>Authors</FormLabel>
              <DynamicInput onChange={handleAuthorsChange} />

              <FormLabel>Search in title</FormLabel>
              <Input type="text" onChange={handleTitleChange} />
              <FormLabel>Search in content</FormLabel>
              <Input type="text" onChange={handleContentChange} />
            </FormControl>
          </DrawerBody>

          <DrawerFooter>
            <Button colorScheme="red" variant="outline" mr={3}>
              Reset
            </Button>
            <Button variant="outline" mr={3} onClick={onClose}>
              Cancel
            </Button>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
    </>
  );
}

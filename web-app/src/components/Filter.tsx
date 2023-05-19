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
  FormErrorMessage,
  FormHelperText,
  HStack,
  VStack,
  Flex,
  Heading,
  useDisclosure,
} from "@chakra-ui/react";
import { useState, useEffect, useRef } from "react";
import DynamicInput from "./DynamicInput";

export default function Filter() {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const btnRef = React.useRef<HTMLButtonElement>(null);

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
              <FormLabel>Date</FormLabel>
              <HStack>
                <Input type="date" />
                <Input type="date" />
              </HStack>
              <FormLabel>Categories</FormLabel>
              <DynamicInput />
              <FormLabel>Authors</FormLabel>
              <DynamicInput />

              <FormLabel>Search in title</FormLabel>
              <Input type="text" />
              <FormLabel>Search in content</FormLabel>
              <Input type="text" />
            </FormControl>
          </DrawerBody>

          <DrawerFooter>
            <Button colorScheme="red" variant="outline" mr={3}>
              Reset
            </Button>
            <Button variant="outline" mr={3} onClick={onClose}>
              Cancel
            </Button>
            <Button colorScheme="blue" onClick={onClose}>
              Save
            </Button>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
    </>
  );
}

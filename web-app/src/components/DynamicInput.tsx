import React from "react";
import { useState, useEffect, useRef } from "react";
import { Flex, Input, Button } from "@chakra-ui/react";

export default function DynamicInput() {
  const [inputs, setInputs] = useState<string[]>([]);
  const add = () => {
    setInputs([...inputs, ""]);
  };

  const remove = (index: number) => {
    const updated = [...inputs];
    updated.splice(index, 1);
    setInputs(updated);
  };

  const change = (value: string, index: number) => {
    const updated = [...inputs];
    updated[index] = value;
    setInputs(updated);
  };

  return (
    <Flex direction="column">
      {inputs.map((input, index) => (
        <Flex key={index} mb={2}>
          <Input
            value={input}
            onChange={(e) => change(e.target.value, index)}
            placeholder="Enter a value"
          />
          <Button ml={2} colorScheme="red" onClick={() => remove(index)}>
            -
          </Button>
        </Flex>
      ))}
      <Button colorScheme="green" onClick={add}>
        +
      </Button>
    </Flex>
  );
}

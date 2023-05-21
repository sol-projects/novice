import React from "react";
import { useState, useEffect } from "react";
import { Flex, Input, Button } from "@chakra-ui/react";

interface DynamicInputProps {
  onChange: (values: string[]) => void;
}

export default function DynamicInput({ onChange }: DynamicInputProps) {
  const [inputs, setInputs] = useState<string[]>([]);

  const add = () => {
    setInputs([...inputs, ""]);
  };

  const remove = (index: number) => {
    const updated = [...inputs];
    updated.splice(index, 1);
    setInputs(updated);
    onChange(updated);
  };

  const change = (value: string, index: number) => {
    const updated = [...inputs];
    updated[index] = value;
    setInputs(updated);
    onChange(updated);
  };

  useEffect(() => {
    onChange(inputs);
  }, [inputs, onChange]);

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

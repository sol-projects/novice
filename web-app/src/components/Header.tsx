import React from "react";
import {
  Box,
  Link,
  Container,
  Flex,
  HStack,
  IconButton,
  useBreakpointValue,
} from "@chakra-ui/react";
import { Link as RouterLink } from "react-router-dom";

export default function Header() {
  return (
    <Box as="section" pb={{ base: "12", md: "24" }}>
      <Box as="nav" bg="bg-surface" boxShadow="sm">
        <Container py={{ base: "4", lg: "5" }}>
          <HStack spacing="10" justify="space-between">
            <Flex justify="space-between" flex="1">
              <Link as={RouterLink} to="/">
                Novice
              </Link>
              <Link as={RouterLink} to="/map">
                Zemljevid
              </Link>
              <Link as={RouterLink} to="/graph">
                Grafi
              </Link>
              <Link as={RouterLink} to="/about">
                O strani
              </Link>
            </Flex>
          </HStack>
        </Container>
      </Box>
    </Box>
  );
}

import React from "react";
import {
  ButtonGroup,
  Container,
  IconButton,
  Stack,
  Text,
} from "@chakra-ui/react";
import { FaGithub } from "react-icons/fa";

export default function Footer() {
  return (
    <Container as="footer" role="contentinfo" py={{ base: "12", md: "16" }}>
      <Stack spacing={{ base: "4", md: "3" }}>
        <Stack justify="space-between" direction="row" align="center">
          <Text fontWeight="bold">SOL</Text>
          <ButtonGroup variant="ghost">
            <IconButton
              as="a"
              href="https://github.com/sol-projects/novice"
              aria-label="GitHub"
              icon={<FaGithub fontSize="1.25rem" />}
            />
          </ButtonGroup>
        </Stack>
        <Text fontSize="sm" color="subtle">
          &copy; {new Date().getFullYear()} SOL
        </Text>
      </Stack>
    </Container>
  );
}

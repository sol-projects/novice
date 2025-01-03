import React from "react";
import {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Text,
  Stack,
  Heading,
  Link,
  Tag,
  Flex,
  Spacer,
  Wrap,
  WrapItem,
} from "@chakra-ui/react";

export default function NewsArticle(props: any) {
  const { article } = props;
  const date = new Date(article.date);
  const displayedTags = article.categories.slice(0, 4);
  const remainingTagsCount = article.categories.length - 4;

  const addView = async () => {
    console.log("success");
    try {
      await fetch("http://localhost:8000/news/views", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ id: article._id }),
      });
    } catch (error) {
      console.error("Error creating view", error);
    }
  };

  return (
    <Link
      href={article.url}
      style={{ textDecoration: "none" }}
      onClick={addView}
      isExternal
    >
      <Card
        height="100%"
        width="100%"
        shadow="md"
        direction={{ base: "column", sm: "row" }}
        overflow="hidden"
        variant="outline"
        padding="1%"
        transition="0.15s all ease 0s"
        _hover={{
          transform: "translateY(-2px)",
          boxShadow: "xl",
        }}
      >
        <Stack width="30%">
          <Text>
            Stran: {new URL(article.url).hostname.replace("www.", "")}
          </Text>
          <Text>Avtorji: {article.authors.toString()}</Text>
          <Text>Datum: {date.toLocaleString("sl-SI")}</Text>
          <Text>Ogledi: {article.views.length}</Text>
          <Wrap mt={2} spacing={2}>
            {displayedTags.map((tag: string, index: number) => (
              <WrapItem key={index}>
                <Tag
                  size="md"
                  variant="solid"
                  colorScheme="teal"
                  borderRadius="full"
                  px={3}
                  py={1}
                >
                  {tag}
                </Tag>
              </WrapItem>
            ))}
            {remainingTagsCount > 0 && (
              <WrapItem>
                <Tag
                  size="md"
                  variant="solid"
                  colorScheme="teal"
                  borderRadius="full"
                  px={3}
                  py={1}
                >
                  +{remainingTagsCount}
                </Tag>
              </WrapItem>
            )}
          </Wrap>
        </Stack>
        <Stack width="70%">
          <CardBody>
            <Heading size="md">{article.title}</Heading>
            <Text align="justify">
              {article.content.length > 500
                ? `${article.content.substring(0, 500)}...`
                : article.content}
            </Text>
          </CardBody>
        </Stack>
      </Card>
    </Link>
  );
}

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
} from "@chakra-ui/react";

export default function NewsArticle(props: any) {
  const { article } = props;
  const date = new Date(article.date);

  const addView = async () => {
    console.log("success")
    try {
      await fetch("http://localhost:8000/news/view", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ newsId: article._id }),
      });
    } catch (error) {
      console.error("Error creating view", error);
    }
  };

  return (
    <Link href={article.url}
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
      >
        <Stack width="30%">
          <Text>
            Stran: {new URL(article.url).hostname.replace("www.", "")}
          </Text>
          <Text>Avtorji: {article.authors.toString()}</Text>
          <Text>Datum: {date.toLocaleString("sl-SI")}</Text>
          <Text>Ogledi: {article.views.length}</Text>
          <Text>Kategorije: {article.categories.toString()}</Text>
        </Stack>
        <Stack width="70%">
          <CardBody>
            <Heading size="md">{article.title}</Heading>
            <Text>
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

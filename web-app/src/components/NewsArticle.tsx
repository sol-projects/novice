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

  return (
    <Link href={article.url} style={{ textDecoration: "none" }}>
      <Card
        height="100%"
        width="100%"
        shadow="md"
        direction={{ base: "column", sm: "row" }}
        overflow="hidden"
        variant="outline"
      >
        <Stack width="30%">
          <Text>Avtorji: {article.authors.toString()}</Text>
          <Text>Datum: {date.toLocaleString("sl-SI")}</Text>
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

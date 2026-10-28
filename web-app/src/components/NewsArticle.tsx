import React from "react";
import INews from "../news/model";
import { FiArrowUpRight, FiEye } from "react-icons/fi";

export default function NewsArticle({ article }: { article: INews }) {
  const date = new Date(article.date);
  const displayedTags = article.categories.slice(0, 4);
  const remainingTagsCount = article.categories.length - 4;

  const addView = async () => {
    try {
      await fetch("http://localhost:8000/news/views", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: article._id }),
      });
    } catch (error) {
      console.error("Error creating view", error);
    }
  };

  return (
    <article className="news-article">
      <a href={article.url} onClick={addView} target="_blank" rel="noopener noreferrer" className="article-link">
        <div className="article-meta">
          <span className="article-source">{new URL(article.url).hostname.replace("www.", "")}</span>
          <time dateTime={date.toISOString()}>{date.toLocaleString("sl-SI", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" })}</time>
        </div>
        <div className="article-heading"><h2>{article.title}</h2><FiArrowUpRight aria-hidden="true" /></div>
        <p className="article-excerpt">{article.content.length > 500 ? `${article.content.substring(0, 500)}…` : article.content}</p>
        <div className="article-bottom">
          <span className="article-authors">{article.authors.join(", ") || "Avtor ni naveden"}</span>
          <span className="article-tags">{displayedTags.map((tag, index) => <span className="tag" key={index}>{tag}</span>)}{remainingTagsCount > 0 && <span className="tag">+{remainingTagsCount}</span>}</span>
          <span className="article-views"><FiEye aria-hidden="true" /><span>{article.views.length}<span className="visually-hidden"> ogledov</span></span></span>
        </div>
      </a>
    </article>
  );
}

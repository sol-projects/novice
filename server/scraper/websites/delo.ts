import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';
import * as Db from '../../db/db';

async function _delo(n: number) {
  const news: INews[] = [];
  const news_promises: Promise<{
    authors: string[];
    content: string;
    categories: string[];
  }>[] = [];

  const response = await axios.get('https://www.delo.si/zadnje/');
  const $ = cheerio.load(response.data);

  $('.paginator_item').each((i, element) => {
    const title: string | undefined = $(element)
      .find('.article_teaser_timeline__title_text')
      .text();
    if (!title) {
      console.log('Cannot get title');
      return false;
    }
    let url: string | undefined = $(element).find('a').attr('href');
    if (!url) {
      console.log('Cannot get url');
      return false;
    }

    url = `http://www.delo.si${url}`;

    const news_promise = get_newspage(url);
    news_promises.push(news_promise);

    const date_unparsed: string | undefined = $(element)
      .find('.article_teaser_timeline__date_holder')
      .text();
    if (!date_unparsed) {
      console.log('Cannot get date');
      return false;
    }

    const date_split = date_unparsed.split('.');
    const date = new Date(
      +date_split[2],
      +date_split[1] - 1,
      +date_split[0] + 1,
      0,
      0,
      0
    );

    news.push({
      title: title.trim(),
      url: url,
      date: date,
      authors: [],
      content: '',
      categories: [],
      views: [],
      location: {
        type: 'Point',
        coordinates: [0, 0],
      },
    });

    if (i == n - 1) {
      return false;
    }
  });

  const news_responses = await Promise.all(news_promises);

  await Promise.all(
    news_responses.map(async (news_response, i) => {
      const location = Db.Util.getFirstSettlement(news_response.categories);
      const coords = await Db.Util.toCoords(location);
      if (location !== '') {
        news[i].location = { type: 'Point', coordinates: coords };
      }
      news[i].authors = news_response.authors;
      news[i].content = news_response.content;
      news[i].categories = news_response.categories;
    })
  );

  return news;
}

async function get_newspage(
  url: string
): Promise<{ authors: string[]; content: string; categories: string[] }> {
  try {
    const news_response = await axios.get(url);
    const news$ = cheerio.load(news_response.data);

    const authors: string = news$('.article__author_name')
      .first()
      .text()
      .trim();
    if (!authors) {
      console.log(`Cannot fetch author from page: ${url}`);
    }

    const content: string = news$('.article__content')
      .find('p, div')
      .not('.store__links')
      .text()
      .trim();

    if (!content) {
      console.log(`Cannot fetch content from page: ${url}`);
    }

    const categories = news$('.tags__btn[href^="/tag/"]')
      .map((i, el) => news$(el).text().trim().toLowerCase())
      .get();

    return { authors: authors.split(', '), content, categories };
  } catch (error) {
    console.log(`Cannot fetch page ${url}`);
    return { authors: [], content: '', categories: [] };
  }
}

export = _delo;

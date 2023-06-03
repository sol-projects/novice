import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';

async function gov_shared(n: number, website: string) {
  const news: INews[] = [];
  const news_promises: Promise<{ author: string; content: string }>[] = [];

  const response = await axios.get(website);
  const $ = cheerio.load(response.data);

  $('.list-item').each((i, element) => {
    const title: string | undefined = $(element).find('.title').text();
    if (!title) {
      console.log('Cannot get title');
      return false;
    }
    let url: string | undefined = $(element).find('a').attr('href');
    if (!url) {
      console.log('Cannot get url');
      return false;
    }

    url = `http://www.gov.si${url}`;

    const news_promise = get_newspage(url);
    news_promises.push(news_promise);

    const date_unparsed: string | undefined = $(element)
      .find('.datetime time')
      .text();
    if (!date_unparsed) {
      console.log('Cannot get date');
      return false;
    }

    const date_split = date_unparsed.split('. ');
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
  news_responses.forEach((news_response, i) => {
    news[i].authors.push(news_response.author);
    news[i].content = news_response.content;
  });

  return news;
}

async function get_newspage(
  url: string
): Promise<{ author: string; content: string }> {
  try {
    const news_response = await axios.get(url);
    const news$ = cheerio.load(news_response.data);

    const author: string = news$('.organisations')
      .first()
      .find('a')
      .text()
      .trim();
    if (!author) {
      console.log(`Cannot fetch author from page: ${url}`);
    }

    const content: string = news$(
      '.content.col.left.grid-col-8  > :not(:first-child)'
    )
      .text()
      .trim();
    if (!content) {
      console.log(`Cannot fetch content from page: ${url}`);
    }

    return { author, content };
  } catch (error) {
    console.log(`Cannot fetch page ${url}`);
    return { author: '', content: '' };
  }
}

export = gov_shared;

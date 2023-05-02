import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';

async function _24ur(n: number) {
  const news: INews[] = [];
  const news_promises: Promise<{
    authors: string[];
    date: Date;
    location: string;
    content: string;
  }>[] = [];

  const response = await axios.get('https://www.24ur.com/novice');
  const $ = cheerio.load(response.data);

  $('a:not(:has(div:contains("OGLAS")))').each((i, element) => {
    const href = $(element).attr('href');
    if (href && href.startsWith('/novice/')) {
      let title = $(element).find(
        'div h1, div h2, div h3, div h4, div h5, div h6'
      );

      if (title.text().trim().length > 0) {
        const url = `http://www.24ur.com${href}`;
        const news_promise = get_newspage(url);
        news_promises.push(news_promise);
        const new_news: INews = {
          title: title.text().trim(),
          url: url,
          date: new Date(),
          authors: [],
          content: '',
          image_info: '',
          categories: [],
          location: '',
        };

        news.push(new_news);

        if (news.length == n) {
          return false;
        }
      } else {
        let title2 = $(element)
          .find('h1 span, h2 span, h3 span, h4 span, h5 span, h6 span')
          .text()
          .trim();
        if (title2.length > 0) {
          const url = `http://www.24ur.com${href}`;
          const news_promise = get_newspage(url);
          news_promises.push(news_promise);
          const new_news: INews = {
            title: title2,
            url: url,
            date: new Date(),
            authors: [],
            content: '',
            image_info: '',
            categories: [],
            location: '',
          };

          news.push(new_news);

          if (news.length == n) {
            return false;
          }
        }
      }
    }
  });

  const news_responses = await Promise.all(news_promises);
  news_responses.forEach((news_response, i) => {
    news[i].authors = news_response.authors;
    news[i].content = news_response.content;
    news[i].location = news_response.location;
    news[i].date = news_response.date;
  });

  return news;
}

async function get_newspage(
  url: string
): Promise<{
  authors: string[];
  date: Date;
  location: string;
  content: string;
}> {
  try {
    const response = await axios.get(url);
    const $ = cheerio.load(response.data);

    const authors: string[] = $('.article__author')
      .map((i, el) => $(el).text().trim())
      .get();
    if (!authors.length) {
      console.log(`Cannot fetch author from page: ${url}`);
    }

    const location_date_unparsed: string = $('.leading-caption').text().trim();
    if (!location_date_unparsed) {
      console.log(`Cannot fetch location and date from page: ${url}`);
    } else {
      const location_date = location_date_unparsed.split(', ');
      const date_split = location_date[1].split('. ');
      const date = new Date(
        +date_split[2],
        +date_split[1] - 1,
        +date_split[0] + 1,
        0,
        0,
        0
      );

      const content: string = $('.article__body').text().trim();
      if (!content) {
        console.log(`Cannot fetch content from page: ${url}`);
      }

      return {
        authors: authors,
        date: date,
        location: location_date[0],
        content: content,
      };
    }
  } catch (error) {
    console.log(`Cannot fetch page ${url}\nError: ${error}`);
  }
  return {
    authors: [],
    location: '',
    date: new Date(0, 0, 0, 0, 0, 0),
    content: '',
  };
}

export = _24ur;

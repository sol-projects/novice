import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';

async function _gov_vlada(n: number) {
  const news: INews[] = [];
  try {
    const response = await axios.get(
      'https://www.gov.si/drzavni-organi/vlada/novice/'
    );
    const $ = cheerio.load(response.data);

    $('.list-item').each((i, element) => {
      const title: string | undefined = $(element).find('.title').text();
      if (!title) {
        console.log('Cannot get title');
        return false;
      }

      const url: string | undefined = $(element)
        .find('.title')
        .find('a')
        .attr('href');
      if (!url) {
        console.log('Cannot get url');
        return false;
      }

      const new_news: INews = {
        title: title.trim(),
        url: `http://www.gov.si${url}`,
        date: new Date(),
        author: '',
        content: '',
        image_info: '',
        categories: [],
        location: '',
      };

      news.push(new_news);

      if (i == n - 1) {
        return false;
      }
    });

    return news;
  } catch (error) {
    console.log(error);
  }

  return '';
}

export = _gov_vlada;

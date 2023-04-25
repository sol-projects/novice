import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';

async function _gov(n: number) {
  const news: INews[] = [];
  try {
    const response = await axios.get('https://www.gov.si/novice/');
    const $ = cheerio.load(response.data);

    $('.title').each((i, element) => {
      const title: string = $(element).text();
      const url: string | undefined = $(element).find('a').attr('href');
      if (!url) {
        console.log('Cannot get url');
        return false;
      }

      const new_news: INews = {
        title: title.trim(),
        url: `http://www.gov.si${url}`,
        website: 'gov.si',
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
}

export = _gov;

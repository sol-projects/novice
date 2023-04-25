import cheerio from 'cheerio';
import axios from 'axios';
import { INews } from '../../model/News';

async function _24ur(n: number) {
  const news: INews[] = [];

  try {
    const response = await axios.get('https://www.24ur.com/novice');
    const $ = cheerio.load(response.data);

    $('a:not(:has(div:contains("OGLAS")))').each((i, element) => {
      const href = $(element).attr('href');
      if (href && href.startsWith('/novice/')) {
        let title = $(element).find(
          'div h1, div h2, div h3, div h4, div h5, div h6'
        );
        if (title.text().trim().length > 0) {
          const new_news: INews = {
            title: title.text().trim(),
            url: `http://www.24ur.com${href}`,
            website: '24ur.com',
            date: new Date(),
            author: '',
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
            const new_news: INews = {
              title: title2,
              url: `http://www.24ur.com${href}`,
              website: '24ur.com',
              date: new Date(),
              author: '',
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
  } catch (error) {
    console.log(error);
  }

  return news;
}

export = _24ur;

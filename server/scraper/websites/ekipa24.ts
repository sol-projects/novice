import puppeteer from 'puppeteer';
import { INews } from '../../model/News';

async function ekipaSvet24(n: number = 5) {
  const news: INews[] = [];

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  await page.goto('https://ekipa.svet24.si/');

  const links = await page.$$('.fpNews.lg\\:w-1\\/3');
  for (let i = 0; i < links.length && news.length < n; i++) {
    const link = links[i];

    const titleElement = await link.$('.fpNews.lg\\:w-1\\/3 a[href^="/clanek"]');
    const titleText = await (titleElement
      ? titleElement.evaluate((e) => (e as HTMLElement).innerText.trim())
      : '');
    console.log('Title:', titleText);

    const url = await link.$eval('a[href^="/clanek"]', (e) =>
      e.getAttribute('href')
    );
    console.log('URL:', url);

    if (url) {
      const articlePage = await browser.newPage();
      await articlePage.goto(`https://ekipa.svet24.si${url}`);

      const authors = await articlePage.$$eval('.top-author', (els) =>
      els.map((e) => {
        const text = (e as HTMLElement).innerText.trim();
        const [author] = text.split('\n');
        return author;
      })
    );
    console.log('Authors:', authors);
    
    const dateElement = await articlePage.$('.top-author');
    const dateText = await (dateElement
      ? dateElement.evaluate((e) => {
          const text = (e as HTMLElement).innerText.trim();
          const [, date] = text.split('\n');
          return date;
        })
      : '');
    console.log('Date:', dateText);
    
      const content = await articlePage.$eval(
        'p',
        (e) => (e as HTMLElement).innerText.trim()
      );
      console.log('Content:', content);

      const categoryLinks = await articlePage.$$(
        'a[href^="/iskanje"]'
      );
      const categories = await Promise.all(
        categoryLinks.map((link) =>
          link.evaluate((e) => (e as HTMLElement).innerText.trim())
        )
      );
      console.log('Categories:', categories);

      news.push({
        title: titleText,
        url: `https://ekipa.svet24.si${url}`,
        date: new Date(dateText),
        authors,
        content,
        categories,
        location: 'Slovenija', // Set the location if required
      });

      await articlePage.close();
    }
  }


  return news;
}

export = ekipaSvet24;

import express, { Request, Response } from 'express';
import websites from '../scraper/websites';
import { INews, News } from '../model/News';

async function run_query(res: Response, query: any) {
  try {
    const news = await News.find(query).exec();
    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to retrieve news from MongoDB');
  }
}

export async function id(req: Request, res: Response) {
  const { id } = req.params;

  try {
    const news = await News.findById(id);

    if (!news) {
      return res.status(404).json({ message: `News with id ${id} not found` });
    }

    res.json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Server error');
  }
}

export async function remove(req: Request, res: Response) {
  const id = req.params.id;

  try {
    const deletedNews = await News.findByIdAndDelete(id);

    if (!deletedNews) {
      return res.status(404).send('News not found');
    }

    res.send(`News with ID ${id} deleted`);
  } catch (error) {
    console.error(error);
    res.status(500).send(`Failed to delete news with ID ${id}`);
  }
}

export async function update(req: Request, res: Response) {
  //get URL and find which website it is then call the correct function to fetch new news info
  res.send('/news/update');
}

export async function all(req: Request, res: Response) {
  run_query(res, {});
}

export async function store(req: Request, res: Response) {
  let news: INews[] = [];
  for await (let [key, value] of websites) {
    const result = await value(req.body.n);
    for (let value of result) {
      news.push(value);
    }
  }

  try {
    await News.create(news);
    res.status(201).json(news);
  } catch (error) {
    console.error(error);
    res.status(500).send('Failed to save news to MongoDB');
  }
}

export async function scrape(req: Request, res: Response) {
  let news: INews[] = [];
  const website = await websites.get(req.params.website);

  if (!website) {
    res
      .status(500)
      .send(`Server error. Website with the name ${website} does not exist.`);
  } else {
    const result = await website(+req.params.n);

    for (let value of result) {
      news.push(value);
    }

    res.json(news);
  }
}

export async function scrapeAll(req: Request, res: Response) {
  let news: INews[] = [];

  for await (let [key, value] of websites) {
    const valueResult = await value(+req.params.n);
    for (let value of valueResult) {
      news.push(value);
    }
  }

  res.json(news);
}

export namespace Filter {
  export async function categories(req: Request, res: Response) {
    run_query(res, { categories: { $all: req.params.categories.split(',') } });
  }

  export async function authors(req: Request, res: Response) {
    run_query(res, { authors: { $all: req.params.authors.split(',') } });
  }

  export function location(req: Request, res: Response) {
    run_query(res, { location: req.params.location });
  }

  export function website(req: Request, res: Response) {
    run_query(res, { url: { $regex: req.params.website, $options: 'i' } });
  }

  export function title(req: Request, res: Response) {
    run_query(res, { title: { $regex: req.params.title, $options: 'i' } });
  }

  export function content(req: Request, res: Response) {
    run_query(res, { content: { $regex: req.params.content, $options: 'i' } });
  }

  export namespace Date {
    export function before(req: Request, res: Response) {
      run_query(res, { date: { $lt: req.params.date } });
    }

    export function after(req: Request, res: Response) {
      run_query(res, { date: { $gt: req.params.date } });
    }

    export function range(req: Request, res: Response) {
      run_query(res, {
        date: {
          $gte: req.params.after,
          $lte: req.params.before,
        },
      });
    }
  }
}

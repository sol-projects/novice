import express, { Express, Request, Response } from 'express';
import * as Controller from '../controller/news';
import * as JWT from '../security/jwt';

const router = express.Router();

router.get('/', Controller.all);
router.get('/:id', Controller.id);
router.get('/scrape/:n', Controller.scrapeAll);
router.get('/scrape/:website/:n', Controller.scrape);

router.get('/categories/:categories', Controller.Filter.categories);
router.get('/authors/:authors', Controller.Filter.authors);
router.get('/location/:location', Controller.Filter.location);
router.get('/website/:website', Controller.Filter.website);
router.get('/date/before/:date', Controller.Filter.Date.before);
router.get('/date/after/:date', Controller.Filter.Date.after);
router.get('/date/after/:after/before/:before', Controller.Filter.Date.range);
router.get('/title/:title', Controller.Filter.title);
router.get('/content/:content', Controller.Filter.content);
//nizka prioriteta: dodaj za fromCoords in toCoords

router.put('/:id', JWT.authorization, Controller.update); //funkcijo "update" implementiraj v controller
router.delete('/:id', JWT.authorization, Controller.remove); //funkcijo "remove" implementiraj v controller
router.post('/', JWT.authorization, Controller.post); // funkcijo "post" implementiraj v controller
router.post('/store', JWT.authorization, Controller.store);
router.post('/login', JWT.login);

export = router;

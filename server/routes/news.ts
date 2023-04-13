import express, { Express, Request, Response } from 'express'
const router = express.Router();

router.get("/", (req: Request, res: Response) => {
  res.send("get news");
})

router.get("/:id", (req: Request, res: Response) => {
  res.send("get news");
})

router.get("/category/:category", (req: Request, res: Response) => {
  res.send("get news");
})

module.exports = router;

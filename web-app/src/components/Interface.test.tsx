import React from "react";
import { ChakraProvider } from "@chakra-ui/react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import News from "./News";
import Filter from "./Filter";
import FilterPanel from "./FilterPanel";
import Chart from "./Chart";
import { getAll } from "../news/api";

jest.mock("../news/api", () => ({ getAll: jest.fn() }));
jest.mock("socket.io-client", () => ({ io: () => ({ on: jest.fn(), disconnect: jest.fn() }) }));
jest.mock("@nivo/line", () => ({ ResponsiveLine: () => <div>Črtni graf</div> }));
jest.mock("@nivo/bar", () => ({ ResponsiveBar: () => <div>Stolpčni graf</div> }));
jest.mock("@nivo/pie", () => ({ ResponsivePie: () => <div>Tortni graf</div> }));
jest.mock("@nivo/waffle", () => ({ ResponsiveWaffle: () => <div>Mrežni graf</div> }));

const articles = [
  { _id: "1", title: "Športna novica", content: "Tekma v Ljubljani", url: "https://www.rtvslo.si/sport/1", date: new Date("2026-01-01T10:00:00Z"), authors: ["Ana"], categories: ["šport"], views: [], location: { type: "Point", coordinates: [14.5, 46.05] } },
  { _id: "2", title: "Vremenska novica", content: "Dež v Mariboru", url: "https://www.24ur.com/novice/2", date: new Date("2026-01-02T10:00:00Z"), authors: ["Bojan"], categories: ["vreme"], views: [], location: { type: "Point", coordinates: [15.6, 46.55] } },
];
const api = getAll as jest.Mock;
const renderUI = (node: React.ReactElement) => render(<ChakraProvider>{node}</ChakraProvider>);

beforeAll(() => {
  window.matchMedia = (query: string) => ({ matches: query.includes("min-width: 62em"), media: query, onchange: null, addListener: jest.fn(), removeListener: jest.fn(), addEventListener: jest.fn(), removeEventListener: jest.fn(), dispatchEvent: jest.fn() });
});
beforeEach(() => { api.mockReset(); api.mockResolvedValue(articles); });

test("closing and reopening the shared panel preserves selected filters", async () => {
  const changed = jest.fn();
  renderUI(<FilterPanel onChange={changed} />);
  const summary = screen.getByText("Filtri");
  const panel = summary.closest("details")!;
  await waitFor(() => expect(panel).toHaveAttribute("open"));
  fireEvent.change(screen.getByLabelText("Išči po naslovu"), { target: { value: "Ljubljana" } });
  fireEvent.click(screen.getByLabelText("RTV Slovenija"));
  fireEvent.click(summary);
  expect(panel).not.toHaveAttribute("open");
  fireEvent.click(summary);
  expect(panel).toHaveAttribute("open");
  expect(screen.getByLabelText("Išči po naslovu")).toHaveValue("Ljubljana");
  expect(screen.getByLabelText("RTV Slovenija")).not.toBeChecked();
  expect(changed.mock.calls.at(-1)[0].title).toBe("Ljubljana");
});

test("all date, source, sorting and text filter fields remain wired", async () => {
  const changed = jest.fn();
  renderUI(<Filter onChange={changed} />);
  expect(screen.getAllByRole("checkbox")).toHaveLength(12);
  fireEvent.click(screen.getByLabelText("RTV Slovenija"));
  expect(changed.mock.calls.at(-1)[0].websites).not.toContain("rtvslo");
  fireEvent.change(screen.getByLabelText("Razvrsti po"), { target: { value: "views" } });
  fireEvent.change(screen.getByLabelText("Išči po naslovu"), { target: { value: "Tekma" } });
  fireEvent.change(screen.getByLabelText("Išči po vsebini"), { target: { value: "Ljubljani" } });
  fireEvent.change(screen.getByLabelText("Do datuma"), { target: { value: "2026-01-01" } });
  const data = changed.mock.calls.at(-1)[0];
  expect(data).toMatchObject({ title: "Tekma", content: "Ljubljani", sortBy: "views" });
  expect(data.to.getHours()).toBe(23);
  fireEvent.change(screen.getByLabelText("Do datuma"), { target: { value: "" } });
  expect(Number.isNaN(changed.mock.calls.at(-1)[0].to.getTime())).toBe(false);
});

test("news presets, search and count selection still work", async () => {
  renderUI(<News />);
  await screen.findByText("Športna novica");
  fireEvent.click(screen.getByRole("radio", { name: "Šport" }));
  expect(screen.getByText("Športna novica")).toBeInTheDocument();
  expect(screen.queryByText("Vremenska novica")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("radio", { name: "Vse novice" }));
  fireEvent.change(screen.getByLabelText("Išči po naslovu"), { target: { value: "Vremenska" } });
  expect(screen.queryByText("Športna novica")).not.toBeInTheDocument();
  expect(screen.getByText("Vremenska novica")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Število novic"), { target: { value: "10" } });
  expect(screen.getByLabelText("Število novic")).toHaveValue("10");
});

test("failed and empty archives have distinct honest states", async () => {
  api.mockResolvedValueOnce(undefined);
  const { unmount } = renderUI(<News />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Novic ni bilo mogoče naložiti.");
  unmount();
  api.mockResolvedValueOnce([]);
  renderUI(<News />);
  await screen.findByText("Ni novic za ta izbor.");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

test("chart changes reuse loaded data and all four chart types are available", async () => {
  renderUI(<Chart />);
  await screen.findByText("Črtni graf");
  fireEvent.change(screen.getByLabelText("Vrsta grafa"), { target: { value: "BarChart" } });
  expect(screen.getByText("Stolpčni graf")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Vrsta grafa"), { target: { value: "PieChart" } });
  expect(screen.getByText("Tortni graf")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Vrsta grafa"), { target: { value: "WaffleChart" } });
  expect(screen.getByText("Mrežni graf")).toBeInTheDocument();
  fireEvent.click(screen.getByLabelText("Avtorji", { selector: "input" }));
  fireEvent.change(screen.getByLabelText("Število podatkov"), { target: { value: "0" } });
  expect(screen.getByLabelText("Število podatkov")).toHaveValue(1);
  await waitFor(() => expect(api).toHaveBeenCalledTimes(1));
});

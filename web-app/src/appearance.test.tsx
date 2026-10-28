import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AppearanceProvider, readAppearance } from "./appearance";
import Settings from "./components/Settings";

const renderSettings = () => render(<AppearanceProvider><Settings /></AppearanceProvider>);
const ready = () => waitFor(() => expect(screen.getByDisplayValue("apple")).not.toBeDisabled());

beforeEach(() => {
  localStorage.clear();
  document.querySelectorAll('link[id^="appearance-"]').forEach(link => link.remove());
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.scheme;
});
afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

test("invalid saved settings safely fall back to the original appearance", () => {
  for (const value of ["{broken", "null", '"kitty"', '{"theme":"other","scheme":"other"}']) {
    localStorage.setItem("novice-appearance", value);
    expect(readAppearance()).toEqual({ theme: "almanac", scheme: "original" });
  }
});

test("a theme applies only after it loads, and scheme and theme survive a new visit", async () => {
  const first = renderSettings();
  await ready();
  expect(screen.getByDisplayValue("almanac")).toBeChecked();
  fireEvent.click(screen.getByDisplayValue("apple"));
  expect(screen.getByText("Pripravljamo videz")).toBeInTheDocument();
  expect(document.documentElement.dataset.theme).toBe("almanac");
  await act(async () => { fireEvent.load(document.getElementById("appearance-apple")!); });
  await ready();
  expect(screen.getByDisplayValue("apple")).toBeChecked();
  fireEvent.click(screen.getByDisplayValue("dark"));
  await ready();
  expect(document.documentElement.dataset.scheme).toBe("dark");
  first.unmount();
  renderSettings();
  await ready();
  expect(screen.getByDisplayValue("apple")).toBeChecked();
  expect(screen.getByDisplayValue("dark")).toBeChecked();
});

test("a failed stylesheet preserves the previous theme and lets the user retry", async () => {
  renderSettings();
  await ready();
  fireEvent.click(screen.getByDisplayValue("terminal"));
  await act(async () => { fireEvent.error(document.getElementById("appearance-terminal")!); });
  await ready();
  expect(screen.getByText(/Teme ni bilo mogoče naložiti/)).toBeInTheDocument();
  expect(screen.getByDisplayValue("almanac")).toBeChecked();
  expect(localStorage.getItem("novice-appearance")).toBeNull();
  fireEvent.click(screen.getByDisplayValue("terminal"));
  await act(async () => { fireEvent.load(document.getElementById("appearance-terminal")!); });
  await ready();
  expect(screen.getByDisplayValue("terminal")).toBeChecked();
});

test("blocked storage still allows changing appearance and explains the persistence limit", async () => {
  jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
  jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
  renderSettings();
  await ready();
  fireEvent.click(screen.getByDisplayValue("forest"));
  await ready();
  expect(screen.getByDisplayValue("forest")).toBeChecked();
  expect(document.documentElement.dataset.scheme).toBe("forest");
  expect(screen.getByText(/brskalnik ne more shraniti/)).toBeInTheDocument();
});

test("a stalled download releases the controls and keeps the previous appearance", async () => {
  renderSettings();
  await ready();
  jest.useFakeTimers();
  fireEvent.click(screen.getByDisplayValue("kitty"));
  await act(async () => { jest.advanceTimersByTime(15000); });
  expect(screen.getByDisplayValue("kitty")).not.toBeDisabled();
  expect(screen.getByDisplayValue("almanac")).toBeChecked();
  expect(screen.queryByText("Pripravljamo videz")).not.toBeInTheDocument();
  expect(screen.getByText(/Teme ni bilo mogoče naložiti/)).toBeInTheDocument();
});

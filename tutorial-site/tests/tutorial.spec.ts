import { expect, test } from "@playwright/test";

test("renders the tutorial guide and interactive command rail", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Train, compress, and adapt/i })).toBeVisible();
  await page.getByRole("button", { name: /Chat Data/i }).click();
  await expect(page.getByText("python -m pip install -r requirements-hf.txt")).toBeVisible();
  await expect(page.getByText("Chat quality starts with dialogue format.")).toBeVisible();
});

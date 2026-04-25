import { expect, test, type Page } from "@playwright/test";

async function coloredSectorPixelCounts(page: Page) {
  return page.locator(".sector-raster-canvas").evaluate((canvas) => {
    const raster = canvas as HTMLCanvasElement;
    const context = raster.getContext("2d");
    if (!context) {
      return { blue: 0, red: 0, green: 0 };
    }
    const { data } = context.getImageData(0, 0, raster.width, raster.height);
    let blue = 0;
    let red = 0;
    let green = 0;
    for (let index = 0; index < data.length; index += 4) {
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      if (b > r + 24 && b > g + 8) {
        blue += 1;
      }
      if (r > b + 24 && r > g + 8) {
        red += 1;
      }
      if (g > r + 18 && g > b + 10) {
        green += 1;
      }
    }
    return { blue, red, green };
  });
}

test("loads the case and renders the synchronized panes", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "EBUS Anatomy Correlation" })).toBeVisible();
  const canvas = page.locator(".scene-canvas canvas");
  await expect(canvas).toBeVisible();
  await expect(page.getByRole("img", { name: "Synchronized labeled EBUS sector" })).toBeVisible();
  await expect(page.locator(".sector-pane")).toHaveAttribute("data-sector-source", "volume_masks", { timeout: 20_000 });
  await expect(page.locator(".layer-toggles label").filter({ hasText: "teaching" }).locator("input")).toBeChecked();
  await expect(page.locator(".layer-toggles label").filter({ hasText: "heart" }).locator("input")).toBeChecked();
  await expect(page.locator(".layer-toggles label").filter({ hasText: "context" }).locator("input")).not.toBeChecked();
  await expect(page.locator(".scene-canvas")).toHaveAttribute("data-scope-model", /EBUS_tip\.glb/);
  await expect(page.locator(".scene-canvas")).toHaveAttribute("data-scope-frame", "fan");

  await expect.poll(async () => (await canvas.boundingBox())?.width ?? 0).toBeGreaterThan(300);
  await expect.poll(async () => (await canvas.boundingBox())?.height ?? 0).toBeGreaterThan(300);
  await expect(page.getByText("lymph node").first()).toBeVisible();
});

test("station snaps, layer toggles, and correlated hover states update", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector(".scene-canvas canvas");

  await page.getByLabel("Station snap").selectOption("station_7_node_a::rms");
  await expect(page.locator(".sector-pane h2")).toHaveText("Station 7 Node A");
  await expect(page.locator(".status-strip")).toContainText("Station 7");
  await expect(page.locator(".status-strip")).toContainText("rms");

  const vesselsToggle = page.locator(".layer-toggles label").filter({ hasText: "vessels" });
  await vesselsToggle.click();
  await expect(vesselsToggle.locator("input")).not.toBeChecked();

  const lymphNodeRow = page.locator(".structure-row").filter({ hasText: "lymph node" }).first();
  await lymphNodeRow.hover();
  await expect(lymphNodeRow).toHaveClass(/active/);
});

test("free navigation hides labels for structures outside the current fan slab", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector(".scene-canvas canvas");

  await page.getByLabel("Station snap").selectOption("station_10r_node_b::default");
  const advance = page.locator("label").filter({ hasText: "Advance / retract" }).locator("input");
  await advance.evaluate((element) => {
    const input = element as HTMLInputElement;
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
    valueSetter?.call(input, "20");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await expect(page.locator(".status-strip")).toContainText("20 mm");
  await expect(page.locator(".sector-pane")).toHaveAttribute("data-sector-source", "volume_masks", { timeout: 60_000 });
  await expect(page.locator(".sector-pane")).not.toContainText("Pulmonary Artery");
  await expect(page.locator(".sector-pane")).not.toContainText("Azygous");
});

test("roll rotates the fan basis without moving the external camera basis", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector(".scene-canvas canvas");

  await page.getByLabel("Station snap").selectOption("station_10r_node_a::default");
  const scene = page.locator(".scene-canvas");
  const cameraDepthAtZero = await scene.getAttribute("data-camera-depth-axis");
  const fanDepthAtZero = await scene.getAttribute("data-fan-depth-axis");
  const fanImageAxisAtZero = await scene.getAttribute("data-fan-image-axis");

  const roll = page.locator("label").filter({ hasText: "Roll" }).locator("input");
  await roll.evaluate((element) => {
    const input = element as HTMLInputElement;
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
    valueSetter?.call(input, "45");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await expect(scene).not.toHaveAttribute("data-fan-depth-axis", fanDepthAtZero ?? "");
  await expect(scene).toHaveAttribute("data-fan-image-axis", fanImageAxisAtZero ?? "");
  await expect(scene).toHaveAttribute("data-camera-depth-axis", cameraDepthAtZero ?? "");
});

test("sector image right side follows the cephalic shaft direction", async ({ page }) => {
  await page.goto("/");
  await page.waitForSelector(".scene-canvas canvas");

  await page.getByLabel("Station snap").selectOption("station_10r_node_a::default");
  const imageAxis = await page.locator(".scene-canvas").getAttribute("data-fan-image-axis");
  const superiorComponent = Number((imageAxis ?? "0,0,0").split(",")[1]);

  expect(superiorComponent).toBeGreaterThanOrEqual(0);
  await expect(page.getByText("cephalic")).toBeVisible();
  await expect(page.getByText("caudal")).toBeVisible();
});

test("volume-shaped vessel cuts render as elongated long-axis structures", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");
  await page.waitForSelector(".scene-canvas canvas");

  await page.getByLabel("Station snap").selectOption("station_10r_node_a::default");
  const advance = page.locator("label").filter({ hasText: "Advance / retract" }).locator("input");
  await advance.evaluate((element) => {
    const input = element as HTMLInputElement;
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
    valueSetter?.call(input, "112");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });

  const svc = page.locator('[data-structure-id="superior_vena_cava"]');
  await expect(svc).toBeVisible();
  await expect.poll(async () => Number(await svc.getAttribute("data-contour-count")), { timeout: 60_000 }).toBeGreaterThan(0);
  await expect.poll(async () => Number(await svc.getAttribute("data-shape-aspect"))).toBeGreaterThan(2.2);
  await expect(page.locator('[data-raster-mask-layer="clean-sector"]')).toBeVisible();
  await expect(page.locator(".structure-row").filter({ hasText: "Superior Vena Cava" }).locator(".swatch")).toHaveCSS(
    "background-color",
    "rgb(34, 118, 201)"
  );
  await expect(svc.locator("path").first()).toHaveAttribute("fill", "none");
  await expect.poll(async () => await page.locator(".sector-svg line").count()).toBeLessThan(8);
  const coloredPixelCounts = await coloredSectorPixelCounts(page);
  expect(coloredPixelCounts.blue).toBeGreaterThan(100);
  expect(coloredPixelCounts.red).toBeGreaterThan(100);
});

test("sector keeps the last clean raster while volume masks refresh during scope motion", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");
  await page.waitForSelector(".scene-canvas canvas");

  await page.getByLabel("Station snap").selectOption("station_10r_node_a::default");
  await expect(page.locator(".sector-pane")).toHaveAttribute("data-sector-source", "volume_masks", { timeout: 60_000 });
  await expect.poll(async () => {
    const counts = await coloredSectorPixelCounts(page);
    return counts.blue + counts.red + counts.green;
  }).toBeGreaterThan(100);

  await page.route("**/api/sector-volume?**", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    await route.continue();
  });

  const advance = page.locator("label").filter({ hasText: "Advance / retract" }).locator("input");
  await advance.evaluate((element) => {
    const input = element as HTMLInputElement;
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
    valueSetter?.call(input, "112");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });

  await expect(page.locator(".sector-pane")).toHaveAttribute("data-sector-source", "volume_masks_refreshing", { timeout: 5_000 });
  await expect.poll(async () => {
    const counts = await coloredSectorPixelCounts(page);
    return counts.blue + counts.red + counts.green;
  }).toBeGreaterThan(100);
  await expect(page.locator('[data-raster-mask-layer="clean-sector"]')).toBeVisible();
});

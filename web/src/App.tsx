import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

type Vec3 = [number, number, number];
type Vec2 = [number, number];

type MeshAsset = {
  vertices: Vec3[];
  triangles: Vec3[];
};

type PointCloudAsset = {
  key: string;
  label: string;
  points: Vec3[];
};

type CenterlinePolyline = {
  line_index: number;
  points: Vec3[];
  cumulative_lengths_mm: number[];
  total_length_mm: number;
};

type CenterlineAsset = {
  primary_line_index: number;
  primary_total_length_mm: number;
  polylines: CenterlinePolyline[];
};

type ListedAsset = {
  key: string;
  label: string;
  asset: string;
  color: string;
  point_count: number;
};

type CleanModelAsset = {
  key: string;
  label: string;
  asset: string;
  coordinate_frame: string;
  web_transform: string;
  primary?: boolean;
};

type WebPreset = {
  preset_key: string;
  preset_id: string;
  station: string;
  node: string;
  approach: string;
  label: string;
  line_index: number;
  centerline_s_mm: number;
  contact: Vec3;
  target: Vec3;
  target_lps: Vec3;
  station_key: string;
  vessel_overlays: string[];
  contact_to_target_distance_mm: number;
};

type NodeMarker = {
  key: string;
  preset_key: string;
  station_key: string;
  label: string;
  position: Vec3;
  radius_mm: number;
  color: string;
};

type WebCaseManifest = {
  case_id: string;
  render_defaults: {
    sector_angle_deg: number;
    max_depth_mm: number;
    roll_deg: number;
  };
  bounds: {
    min: Vec3;
    max: Vec3;
    center: Vec3;
    size: Vec3;
  };
  assets: {
    airway_mesh: string;
    centerlines: string;
    vessels: ListedAsset[];
    stations: ListedAsset[];
    clean_models?: CleanModelAsset[];
  };
  navigation: {
    primary_line_index: number;
    primary_total_length_mm: number;
  };
  presets: WebPreset[];
  anatomy: {
    nodes: NodeMarker[];
  };
  color_map: Record<string, string>;
};

type LoadedAssets = {
  airway: MeshAsset;
  centerlines: CenterlineAsset;
  vessels: Record<string, PointCloudAsset>;
  stations: Record<string, PointCloudAsset>;
};

type ProbePose = {
  position: THREE.Vector3;
  tangent: THREE.Vector3;
  depthAxis: THREE.Vector3;
  lateralAxis: THREE.Vector3;
};

type SectorItem = {
  id: string;
  label: string;
  kind: "airway" | "node" | "vessel" | "contact";
  color: string;
  depthMm: number;
  lateralMm: number;
  visible: boolean;
  depthExtentMm?: [number, number];
  lateralExtentMm?: [number, number];
  majorAxisMm?: number;
  minorAxisMm?: number;
  majorAxisVectorMm?: [number, number];
  aspectRatio?: number;
  contoursMm?: Vec2[][];
  contourCount?: number;
  contourSource?: string;
};

type VolumeSectorLabel = {
  id: string;
  key: string;
  label: string;
  kind: "node" | "vessel";
  color: string;
  depth_mm: number;
  lateral_mm: number;
  visible: boolean;
  depth_extent_mm?: [number, number];
  lateral_extent_mm?: [number, number];
  major_axis_mm?: number;
  minor_axis_mm?: number;
  major_axis_vector_mm?: [number, number];
  aspect_ratio?: number;
  contours_mm?: Vec2[][];
  contour_count?: number;
  contour_source?: string;
};

type VolumeSectorResponse = {
  source: "volume_masks";
  sector: {
    labels: VolumeSectorLabel[];
  };
};

type VolumeSectorState = {
  status: "idle" | "loading" | "ready" | "error";
  queryKey: string;
  labels: VolumeSectorLabel[];
};

type LayerState = {
  airway: boolean;
  vessels: boolean;
  heart: boolean;
  nodes: boolean;
  stations: boolean;
  context: boolean;
  centerline: boolean;
  fan: boolean;
};

const DEFAULT_LAYERS: LayerState = {
  airway: true,
  vessels: true,
  heart: true,
  nodes: true,
  stations: true,
  context: false,
  centerline: true,
  fan: true
};
const SECTOR_SLAB_HALF_THICKNESS_MM = 4;
const SNAP_TARGET_SLAB_HALF_THICKNESS_MM = 18;
const WEB_CEPHALIC_AXIS = new THREE.Vector3(0, 1, 0);
const GLB_SCENE_TO_WEB_MM_MATRIX = new THREE.Matrix4().set(
  1000, 0, 0, 0,
  0, 1000, 0, 0,
  0, 0, 1000, 0,
  0, 0, 0, 1
);
const cleanModelCache = new Map<string, Promise<THREE.Group>>();

function toVector(point: Vec3): THREE.Vector3 {
  return new THREE.Vector3(point[0], point[1], point[2]);
}

function toTuple(vector: THREE.Vector3): Vec3 {
  return [vector.x, vector.y, vector.z];
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function formatStation(station: string): string {
  return station.toUpperCase();
}

function primaryCleanModel(caseData: WebCaseManifest): CleanModelAsset | null {
  const models = caseData.assets.clean_models ?? [];
  return models.find((asset) => asset.primary) ?? models[0] ?? null;
}

function cleanModelUrl(asset: CleanModelAsset): string {
  return `/api/asset/${asset.asset}`;
}

function loadCleanModel(asset: CleanModelAsset): Promise<THREE.Group> {
  const url = cleanModelUrl(asset);
  const cached = cleanModelCache.get(url);
  if (cached) {
    return cached;
  }
  const loader = new GLTFLoader();
  const promise = loader.loadAsync(url).then((gltf) => gltf.scene);
  cleanModelCache.set(url, promise);
  return promise;
}

function normalizedAnatomyName(name: string): string {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, " ").replace(/\s+/g, " ").trim();
}

function cleanModelStructureId(name: string): string {
  const normalized = normalizedAnatomyName(name);
  if (!normalized) {
    return "context";
  }
  if (normalized === "airway") {
    return "airway_wall";
  }
  if (normalized.startsWith("station ")) {
    return normalized.replace("station ", "station_").replace(/\s+/g, "").toLowerCase();
  }
  const exact: Record<string, string> = {
    "left atrial appendage": "atrial_appendage_left",
    "azygous vein": "azygous",
    "superior vena cava": "superior_vena_cava"
  };
  return exact[normalized] ?? normalized.replace(/\s+/g, "_");
}

function cleanModelLayer(structureId: string): keyof LayerState {
  if (structureId === "airway_wall") {
    return "airway";
  }
  if (structureId.startsWith("station_")) {
    return "stations";
  }
  if (
    structureId.includes("atrium") ||
    structureId.includes("ventricle") ||
    structureId.includes("appendage")
  ) {
    return "heart";
  }
  if (
    structureId.includes("artery") ||
    structureId.includes("vein") ||
    structureId.includes("venous") ||
    structureId.includes("cava") ||
    structureId.includes("aorta") ||
    structureId.includes("azygous") ||
    structureId.includes("trunk")
  ) {
    return "vessels";
  }
  return "context";
}

function cleanModelColor(structureId: string, colorMap: Record<string, string>): string {
  if (structureId === "airway_wall") {
    return colorMap.airway ?? "#22c7c9";
  }
  if (structureId.startsWith("station_")) {
    return colorMap.lymph_node ?? colorMap.station ?? "#93c56f";
  }
  if (colorMap[structureId]) {
    return colorMap[structureId];
  }
  if (structureId.includes("artery") || structureId.includes("aorta") || structureId.includes("trunk")) {
    return "#d13f3f";
  }
  if (structureId.includes("vein") || structureId.includes("venous") || structureId.includes("cava")) {
    return "#2276c9";
  }
  if (structureId.includes("azygous")) {
    return "#2276c9";
  }
  if (structureId.includes("atrium") || structureId.includes("appendage")) {
    return "#d13f3f";
  }
  if (structureId.includes("ventricle")) {
    return structureId.includes("right") ? "#2276c9" : "#d13f3f";
  }
  if (structureId.includes("esophagus")) {
    return "#b79667";
  }
  return "#7b8587";
}

function cleanModelOpacity(layer: keyof LayerState, highlighted: boolean, teachingView: boolean): number {
  if (teachingView && !highlighted) {
    const teachingOpacityByLayer: Partial<Record<keyof LayerState, number>> = {
      airway: 0.1,
      vessels: 0.12,
      heart: 0.22,
      stations: 0.08,
      context: 0.04
    };
    return teachingOpacityByLayer[layer] ?? 0.06;
  }
  if (highlighted) {
    if (layer === "airway") {
      return 0.22;
    }
    if (layer === "heart") {
      return 0.62;
    }
    return layer === "context" ? 0.42 : 0.74;
  }
  const opacityByLayer: Partial<Record<keyof LayerState, number>> = {
    airway: 0.16,
    vessels: 0.58,
    heart: 0.48,
    stations: 0.36,
    context: 0.16
  };
  return opacityByLayer[layer] ?? 0.5;
}

function isTeachingFocus(
  structureId: string,
  selectedPreset: WebPreset,
  intersectedStructureIds: Set<string>,
  activeStructure: string | null
): boolean {
  return (
    structureId === "airway_wall" ||
    structureId === selectedPreset.station_key ||
    activeStructure === structureId ||
    intersectedStructureIds.has(structureId)
  );
}

function disposeMaterial(material: THREE.Material | THREE.Material[] | undefined): void {
  if (Array.isArray(material)) {
    for (const item of material) {
      item.dispose();
    }
    return;
  }
  material?.dispose();
}

function pointAtS(polyline: CenterlinePolyline, sMm: number): THREE.Vector3 {
  const points = polyline.points;
  const lengths = polyline.cumulative_lengths_mm;
  if (points.length === 0) {
    return new THREE.Vector3();
  }
  if (points.length === 1) {
    return toVector(points[0]);
  }
  const clamped = Math.min(Math.max(sMm, 0), polyline.total_length_mm);
  if (clamped <= 0) {
    return toVector(points[0]);
  }
  if (clamped >= polyline.total_length_mm) {
    return toVector(points[points.length - 1]);
  }

  let segmentIndex = 0;
  while (segmentIndex < lengths.length - 1 && lengths[segmentIndex + 1] < clamped) {
    segmentIndex += 1;
  }
  const startLength = lengths[segmentIndex] ?? 0;
  const endLength = lengths[segmentIndex + 1] ?? startLength;
  const t = endLength > startLength ? (clamped - startLength) / (endLength - startLength) : 0;
  return toVector(points[segmentIndex]).lerp(toVector(points[segmentIndex + 1]), t);
}

function tangentAtS(polyline: CenterlinePolyline, sMm: number): THREE.Vector3 {
  const windowMm = 5;
  const start = pointAtS(polyline, Math.max(0, sMm - windowMm));
  const end = pointAtS(polyline, Math.min(polyline.total_length_mm, sMm + windowMm));
  const tangent = end.sub(start);
  if (tangent.lengthSq() > 1e-8) {
    return tangent.normalize();
  }
  if (polyline.points.length >= 2) {
    return toVector(polyline.points[polyline.points.length - 1]).sub(toVector(polyline.points[0])).normalize();
  }
  return new THREE.Vector3(0, 1, 0);
}

function fallbackDepthAxis(tangent: THREE.Vector3): THREE.Vector3 {
  const candidates = [
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, 1)
  ].sort((a, b) => Math.abs(a.dot(tangent)) - Math.abs(b.dot(tangent)));
  for (const candidate of candidates) {
    const projected = candidate.clone().sub(tangent.clone().multiplyScalar(candidate.dot(tangent)));
    if (projected.lengthSq() > 1e-8) {
      return projected.normalize();
    }
  }
  return new THREE.Vector3(0, 1, 0);
}

function computePose(polyline: CenterlinePolyline, sMm: number, rollDeg: number, target: Vec3): ProbePose {
  const position = pointAtS(polyline, sMm);
  const tangent = tangentAtS(polyline, sMm);
  let depthAxis = toVector(target).sub(position);
  depthAxis.sub(tangent.clone().multiplyScalar(depthAxis.dot(tangent)));
  if (depthAxis.lengthSq() <= 1e-8) {
    depthAxis = fallbackDepthAxis(tangent);
  } else {
    depthAxis.normalize();
  }
  depthAxis.applyAxisAngle(tangent, THREE.MathUtils.degToRad(rollDeg)).normalize();
  const lateralAxis = new THREE.Vector3().crossVectors(tangent, depthAxis).normalize();
  depthAxis = new THREE.Vector3().crossVectors(lateralAxis, tangent).normalize();
  return { position, tangent, depthAxis, lateralAxis };
}

function cephalicImageAxis(pose: ProbePose): THREE.Vector3 {
  const axis = pose.tangent.clone().normalize();
  return axis.dot(WEB_CEPHALIC_AXIS) >= 0 ? axis : axis.multiplyScalar(-1);
}

function sectorPlaneNormal(pose: ProbePose): THREE.Vector3 {
  const imageAxis = cephalicImageAxis(pose);
  return new THREE.Vector3().crossVectors(imageAxis, pose.depthAxis).normalize();
}

function projectToSector(
  point: Vec3,
  pose: ProbePose,
  maxDepthMm: number,
  sectorAngleDeg: number,
  slabHalfThicknessMm = SECTOR_SLAB_HALF_THICKNESS_MM
) {
  const offset = toVector(point).sub(pose.position);
  const imageAxis = cephalicImageAxis(pose);
  const planeNormal = sectorPlaneNormal(pose);
  const depthMm = offset.dot(pose.depthAxis);
  const lateralMm = offset.dot(imageAxis);
  const outOfPlaneMm = offset.dot(planeNormal);
  const halfWidth = Math.max(0, depthMm) * Math.tan(THREE.MathUtils.degToRad(sectorAngleDeg / 2));
  const inSectorPlane = depthMm >= 0 && depthMm <= maxDepthMm && Math.abs(lateralMm) <= halfWidth + 1e-6;
  const inSliceSlab = Math.abs(outOfPlaneMm) <= slabHalfThicknessMm;
  return { depthMm, lateralMm, outOfPlaneMm, visible: inSectorPlane && inSliceSlab };
}

function nearestVisiblePoint(points: Vec3[], pose: ProbePose, maxDepthMm: number, sectorAngleDeg: number) {
  let best: { depthMm: number; lateralMm: number; outOfPlaneMm: number; visible: boolean; score: number } | null = null;
  const stride = Math.max(1, Math.floor(points.length / 360));
  for (let index = 0; index < points.length; index += stride) {
    const projected = projectToSector(points[index], pose, maxDepthMm, sectorAngleDeg);
    if (!projected.visible) {
      continue;
    }
    const score =
      Math.abs(projected.outOfPlaneMm) * 2 +
      Math.abs(projected.lateralMm) +
      Math.abs(projected.depthMm - maxDepthMm * 0.55) * 0.25;
    if (best === null || score < best.score) {
      best = { ...projected, score };
    }
  }
  return best;
}

function volumeSectorQueryKey(selectedPreset: WebPreset, lineIndex: number, sMm: number, rollDeg: number, caseData: WebCaseManifest) {
  return [
    selectedPreset.preset_key,
    lineIndex,
    sMm.toFixed(1),
    rollDeg.toFixed(1),
    caseData.render_defaults.max_depth_mm.toFixed(1),
    caseData.render_defaults.sector_angle_deg.toFixed(1)
  ].join("|");
}

function volumeLabelToSectorItem(label: VolumeSectorLabel): SectorItem {
  return {
    id: label.id,
    label: label.label,
    kind: label.kind,
    color: label.color,
    depthMm: label.depth_mm,
    lateralMm: label.lateral_mm,
    visible: label.visible,
    depthExtentMm: label.depth_extent_mm,
    lateralExtentMm: label.lateral_extent_mm,
    majorAxisMm: label.major_axis_mm,
    minorAxisMm: label.minor_axis_mm,
    majorAxisVectorMm: label.major_axis_vector_mm,
    aspectRatio: label.aspect_ratio,
    contoursMm: label.contours_mm,
    contourCount: label.contour_count,
    contourSource: label.contour_source
  };
}

function useCaseData() {
  const [caseData, setCaseData] = useState<WebCaseManifest | null>(null);
  const [assets, setAssets] = useState<LoadedAssets | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchJson<T>(path: string): Promise<T> {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}: ${path}`);
      }
      return (await response.json()) as T;
    }

    async function load() {
      try {
        const manifest = await fetchJson<WebCaseManifest>("/api/case");
        const [airway, centerlines] = await Promise.all([
          fetchJson<MeshAsset>(`/api/asset/${manifest.assets.airway_mesh}`),
          fetchJson<CenterlineAsset>(`/api/asset/${manifest.assets.centerlines}`)
        ]);
        const vesselEntries = await Promise.all(
          manifest.assets.vessels.map(async (asset) => [asset.key, await fetchJson<PointCloudAsset>(`/api/asset/${asset.asset}`)] as const)
        );
        const stationEntries = await Promise.all(
          manifest.assets.stations.map(async (asset) => [asset.key, await fetchJson<PointCloudAsset>(`/api/asset/${asset.asset}`)] as const)
        );
        if (!cancelled) {
          setCaseData(manifest);
          setAssets({
            airway,
            centerlines,
            vessels: Object.fromEntries(vesselEntries),
            stations: Object.fromEntries(stationEntries)
          });
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : String(loadError));
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { caseData, assets, error };
}

function AnatomyScene({
  caseData,
  assets,
  pose,
  cameraPose,
  selectedPreset,
  layers,
  teachingView,
  activeStructure,
  intersectedStructureIds
}: {
  caseData: WebCaseManifest;
  assets: LoadedAssets;
  pose: ProbePose;
  cameraPose: ProbePose;
  selectedPreset: WebPreset;
  layers: LayerState;
  teachingView: boolean;
  activeStructure: string | null;
  intersectedStructureIds: Set<string>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.setClearColor("#101416", 1);
    container.replaceChildren(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog("#101416", 360, 780);

    const boundsCenter = toVector(caseData.bounds.center);
    const size = toVector(caseData.bounds.size);
    const sceneRadius = Math.max(size.x, size.y, size.z, 180);
    const focus = cameraPose.position.clone().add(cameraPose.depthAxis.clone().multiplyScalar(15));
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, sceneRadius * 8);
    camera.position.copy(
      focus
        .clone()
        .add(cameraPose.lateralAxis.clone().multiplyScalar(92))
        .add(cameraPose.depthAxis.clone().multiplyScalar(-118))
        .add(cameraPose.tangent.clone().multiplyScalar(58))
        .add(new THREE.Vector3(0, 54, 0))
    );
    camera.lookAt(focus);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(focus);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.8;

    scene.add(new THREE.AmbientLight("#f2f7ef", 1.75));
    const light = new THREE.DirectionalLight("#ffffff", 2);
    light.position.copy(boundsCenter.clone().add(new THREE.Vector3(100, 160, 120)));
    scene.add(light);

    const cleanModel = primaryCleanModel(caseData);
    let cancelled = false;

    if (cleanModel) {
      loadCleanModel(cleanModel)
        .then((template) => {
          if (cancelled) {
            return;
          }
          const model = template.clone(true);
          model.name = `clean-model:${cleanModel.key}`;
          model.applyMatrix4(GLB_SCENE_TO_WEB_MM_MATRIX);
          let visibleMeshCount = 0;
          model.traverse((object) => {
            const mesh = object as THREE.Mesh;
            if (!mesh.isMesh) {
              return;
            }
            const structureId = cleanModelStructureId(mesh.name || mesh.parent?.name || "");
            const layer = cleanModelLayer(structureId);
            mesh.visible = Boolean(layers[layer]);
            if (!mesh.visible) {
              return;
            }
            visibleMeshCount += 1;
            if (!mesh.geometry.getAttribute("normal")) {
              mesh.geometry.computeVertexNormals();
            }

            const highlighted = isTeachingFocus(structureId, selectedPreset, intersectedStructureIds, activeStructure);
            mesh.userData.sharedAssetGeometry = true;
            mesh.material = new THREE.MeshBasicMaterial({
              color: cleanModelColor(structureId, caseData.color_map),
              transparent: true,
              opacity: cleanModelOpacity(layer, highlighted, teachingView),
              side: THREE.DoubleSide,
              depthWrite: false
            });
            mesh.userData.generatedMaterial = true;
            mesh.renderOrder = layer === "airway" || layer === "context" ? 0 : 1;
          });
          scene.add(model);
          const modelBounds = new THREE.Box3().setFromObject(model);
          container.dataset.cleanMeshCount = String(visibleMeshCount);
          container.dataset.cleanModelBounds = [
            modelBounds.min.x,
            modelBounds.min.y,
            modelBounds.min.z,
            modelBounds.max.x,
            modelBounds.max.y,
            modelBounds.max.z
          ].map((value) => value.toFixed(1)).join(",");
        })
        .catch((loadError) => {
          console.error("Failed to load clean anatomy model", loadError);
        });
    }

    if (!cleanModel && layers.airway) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(assets.airway.vertices.flat(), 3));
      geometry.setIndex(assets.airway.triangles.flat());
      geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({
        color: caseData.color_map.airway ?? "#22c7c9",
        transparent: true,
        opacity: 0.34,
        roughness: 0.66,
        metalness: 0.02,
        side: THREE.DoubleSide
      });
      scene.add(new THREE.Mesh(geometry, material));
    }

    if (layers.centerline) {
      const group = new THREE.Group();
      for (const polyline of assets.centerlines.polylines) {
        const geometry = new THREE.BufferGeometry().setFromPoints(polyline.points.map(toVector));
        const material = new THREE.LineBasicMaterial({
          color: polyline.line_index === selectedPreset.line_index ? "#eef4f2" : "#56666a",
          transparent: true,
          opacity: polyline.line_index === selectedPreset.line_index ? 0.82 : 0.28
        });
        group.add(new THREE.Line(geometry, material));
      }
      scene.add(group);
    }

    if (!cleanModel && layers.stations) {
      for (const station of caseData.assets.stations) {
        const points = assets.stations[station.key]?.points ?? [];
        if (!points.length) {
          continue;
        }
        const geometry = new THREE.BufferGeometry().setFromPoints(points.map(toVector));
        const highlighted = isTeachingFocus(station.key, selectedPreset, intersectedStructureIds, activeStructure);
        const material = new THREE.PointsMaterial({
          color: station.color,
          size: highlighted ? 2.5 : 1.55,
          transparent: true,
          opacity: highlighted ? 0.9 : teachingView ? 0.1 : 0.48,
          depthWrite: false
        });
        scene.add(new THREE.Points(geometry, material));
      }
    }

    if (!cleanModel && layers.vessels) {
      for (const vessel of caseData.assets.vessels) {
        const points = assets.vessels[vessel.key]?.points ?? [];
        if (!points.length) {
          continue;
        }
        const geometry = new THREE.BufferGeometry().setFromPoints(points.map(toVector));
        const highlighted = isTeachingFocus(vessel.key, selectedPreset, intersectedStructureIds, activeStructure);
        const material = new THREE.PointsMaterial({
          color: vessel.color,
          size: highlighted ? 2.25 : 1.2,
          transparent: true,
          opacity: highlighted ? 0.9 : teachingView ? 0.12 : 0.4,
          depthWrite: false
        });
        scene.add(new THREE.Points(geometry, material));
      }
    }

    if (layers.nodes) {
      for (const node of caseData.anatomy.nodes) {
        const active = isTeachingFocus(node.station_key, selectedPreset, intersectedStructureIds, activeStructure) || activeStructure === node.key;
        const geometry = new THREE.SphereGeometry(active ? node.radius_mm * 1.1 : node.radius_mm, 18, 12);
        const material = new THREE.MeshStandardMaterial({
          color: node.color,
          transparent: true,
          opacity: active ? 0.94 : teachingView ? 0.08 : 0.46,
          roughness: 0.5
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.copy(toVector(node.position));
        scene.add(sphere);
      }
    }

    const scopeGroup = new THREE.Group();
    const shaftStart = pose.position.clone().add(pose.tangent.clone().multiplyScalar(-22));
    const shaftEnd = pose.position.clone().add(pose.tangent.clone().multiplyScalar(34));
    const shaftGeometry = new THREE.BufferGeometry().setFromPoints([shaftStart, shaftEnd]);
    scopeGroup.add(new THREE.Line(shaftGeometry, new THREE.LineBasicMaterial({ color: "#f5e4c8", linewidth: 2 })));
    const contact = new THREE.Mesh(
      new THREE.SphereGeometry(3.1, 20, 12),
      new THREE.MeshStandardMaterial({ color: "#f5e166", emissive: "#3a2e05", emissiveIntensity: 0.35 })
    );
    contact.position.copy(pose.position);
    scopeGroup.add(contact);
    scene.add(scopeGroup);

    if (layers.fan) {
      const maxDepth = caseData.render_defaults.max_depth_mm;
      const halfWidth = maxDepth * Math.tan(THREE.MathUtils.degToRad(caseData.render_defaults.sector_angle_deg / 2));
      const imageAxis = cephalicImageAxis(pose);
      const apex = pose.position;
      const farCenter = apex.clone().add(pose.depthAxis.clone().multiplyScalar(maxDepth));
      const left = farCenter.clone().add(imageAxis.clone().multiplyScalar(-halfWidth));
      const right = farCenter.clone().add(imageAxis.clone().multiplyScalar(halfWidth));
      const fanGeometry = new THREE.BufferGeometry().setFromPoints([apex, left, right]);
      fanGeometry.setIndex([0, 1, 2]);
      fanGeometry.computeVertexNormals();
      const fan = new THREE.Mesh(
        fanGeometry,
        new THREE.MeshBasicMaterial({
          color: "#8bd4ff",
          transparent: true,
          opacity: 0.18,
          side: THREE.DoubleSide,
          depthWrite: false
        })
      );
      fan.renderOrder = 5;
      scene.add(fan);
      const edgeGeometry = new THREE.BufferGeometry().setFromPoints([apex, left, right, apex]);
      const fanEdge = new THREE.Line(edgeGeometry, new THREE.LineBasicMaterial({ color: "#bfe7ff", transparent: true, opacity: 0.72 }));
      fanEdge.renderOrder = 6;
      scene.add(fanEdge);
    }

    let frameId = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      frameId = window.requestAnimationFrame(render);
    };
    render();

    const resizeObserver = new ResizeObserver(() => {
      const nextWidth = container.clientWidth || width;
      const nextHeight = container.clientHeight || height;
      renderer.setSize(nextWidth, nextHeight);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(container);

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        const mesh = object as THREE.Mesh;
        if (mesh.geometry && !mesh.userData.sharedAssetGeometry) {
          mesh.geometry.dispose();
        }
        if (mesh.userData.generatedMaterial) {
          disposeMaterial(mesh.material);
        }
      });
    };
  }, [activeStructure, assets, cameraPose, caseData, intersectedStructureIds, layers, pose, selectedPreset, teachingView]);

  return (
    <div
      className="scene-canvas"
      ref={containerRef}
      data-camera-depth-axis={cameraPose.depthAxis.toArray().map((value) => value.toFixed(3)).join(",")}
      data-fan-depth-axis={pose.depthAxis.toArray().map((value) => value.toFixed(3)).join(",")}
      data-fan-image-axis={cephalicImageAxis(pose).toArray().map((value) => value.toFixed(3)).join(",")}
      data-clean-model={primaryCleanModel(caseData)?.asset ?? ""}
    />
  );
}

function SectorView({
  items,
  source,
  selectedPreset,
  caseData,
  activeStructure,
  setActiveStructure
}: {
  items: SectorItem[];
  source: string;
  selectedPreset: WebPreset;
  caseData: WebCaseManifest;
  activeStructure: string | null;
  setActiveStructure: (value: string | null) => void;
}) {
  const maxDepth = caseData.render_defaults.max_depth_mm;
  const halfTan = Math.tan(THREE.MathUtils.degToRad(caseData.render_defaults.sector_angle_deg / 2));
  const visibleItems = items.filter((item) => item.visible || item.kind === "airway" || item.kind === "contact");

  function itemPosition(item: SectorItem) {
    const depthRatio = Math.min(Math.max(item.depthMm / maxDepth, 0), 1);
    const x = 50 + (item.lateralMm / Math.max(maxDepth * halfTan, 1)) * 39;
    const y = 8 + depthRatio * 82;
    return { x: Math.min(Math.max(x, 9), 91), y };
  }

  function sectorPoint(point: Vec2) {
    const [lateralMm, depthMm] = point;
    const depthRatio = Math.min(Math.max(depthMm / maxDepth, 0), 1);
    const x = 50 + (lateralMm / Math.max(maxDepth * halfTan, 1)) * 39;
    const y = 8 + depthRatio * 82;
    return { x: clamp(x, 8, 92), y: clamp(y, 7, 93) };
  }

  function contourPath(points: Vec2[]) {
    if (points.length < 2) {
      return "";
    }
    const mapped = points.map(sectorPoint);
    const first = mapped[0];
    const commands = mapped.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)} ${point.y.toFixed(2)}`);
    const last = mapped[mapped.length - 1];
    const closed = Math.hypot(first.x - last.x, first.y - last.y) < 1.6;
    return `${commands.join(" ")}${closed ? " Z" : ""}`;
  }

  function itemShape(item: SectorItem) {
    const xScale = 39 / Math.max(maxDepth * halfTan, 1);
    const yScale = 82 / Math.max(maxDepth, 1);
    let vector: [number, number] = item.majorAxisVectorMm ?? [1, 0];
    if (!item.majorAxisMm && item.lateralExtentMm && item.depthExtentMm) {
      const lateralSpan = item.lateralExtentMm[1] - item.lateralExtentMm[0];
      const depthSpan = item.depthExtentMm[1] - item.depthExtentMm[0];
      vector = Math.abs(lateralSpan) >= Math.abs(depthSpan) ? [1, 0] : [0, 1];
    }
    const vectorNorm = Math.hypot(vector[0], vector[1]) || 1;
    const unitLateral = vector[0] / vectorNorm;
    const unitDepth = vector[1] / vectorNorm;
    const lateralSpan = item.lateralExtentMm ? Math.abs(item.lateralExtentMm[1] - item.lateralExtentMm[0]) : 0;
    const depthSpan = item.depthExtentMm ? Math.abs(item.depthExtentMm[1] - item.depthExtentMm[0]) : 0;
    const majorMm = Math.max(item.majorAxisMm ?? Math.max(lateralSpan, depthSpan), item.kind === "vessel" ? 4.5 : 6);
    const minorMm = Math.max(item.minorAxisMm ?? Math.min(lateralSpan || majorMm, depthSpan || majorMm), item.kind === "vessel" ? 2.5 : 4);
    const majorScale = Math.hypot(unitLateral * xScale, unitDepth * yScale);
    const minorScale = Math.hypot(-unitDepth * xScale, unitLateral * yScale);
    const rawMajor = (majorMm / 2) * majorScale;
    const rawMinor = (minorMm / 2) * minorScale;
    const angleDeg = THREE.MathUtils.radToDeg(Math.atan2(unitDepth * yScale, unitLateral * xScale));

    if (item.kind === "vessel") {
      const rx = clamp(rawMajor, 4.8, 31);
      const ry = clamp(Math.min(rawMinor, rx * 0.3), 2.3, 9.5);
      return {
        rx,
        ry,
        angleDeg,
        displayAspect: rx / Math.max(ry, 0.1)
      };
    }

    const rx = clamp(rawMajor, 6.4, 10.5);
    const ry = clamp(rawMinor, 3.8, 7.0);
    return {
      rx,
      ry,
      angleDeg,
      displayAspect: rx / Math.max(ry, 0.1)
    };
  }

  return (
    <section className="sector-pane" aria-label="Labeled EBUS sector" data-sector-source={source}>
      <div className="pane-header">
        <div>
          <span className="eyebrow">EBUS sector</span>
          <h2>Station {formatStation(selectedPreset.station)} Node {selectedPreset.node.toUpperCase()}</h2>
        </div>
        <span className="approach-chip">{selectedPreset.approach}</span>
      </div>
      <svg className="sector-svg" viewBox="0 0 100 100" role="img" aria-label="Synchronized labeled EBUS sector">
        <defs>
          <radialGradient id="sectorNoise" cx="50%" cy="20%" r="86%">
            <stop offset="0%" stopColor="#2f3435" />
            <stop offset="58%" stopColor="#171b1d" />
            <stop offset="100%" stopColor="#080a0b" />
          </radialGradient>
          <clipPath id="fanClip">
            <path d="M50 7 L10 92 Q50 99 90 92 Z" />
          </clipPath>
        </defs>
        <rect width="100" height="100" fill="#050607" />
        <path d="M50 7 L10 92 Q50 99 90 92 Z" fill="url(#sectorNoise)" stroke="#c6d1d5" strokeWidth="0.55" />
        <g clipPath="url(#fanClip)">
          <path d="M14 14 C27 22, 72 22, 86 14" stroke="#e7e4dd" strokeOpacity="0.56" strokeWidth="1.8" fill="none" />
          <path d="M10 53 C28 45, 72 45, 90 53" stroke="#3a4143" strokeWidth="0.5" fill="none" />
          <path d="M14 72 C34 64, 66 64, 86 72" stroke="#2b3132" strokeWidth="0.5" fill="none" />
          {Array.from({ length: 44 }).map((_, index) => (
            <line
              key={index}
              x1={10 + ((index * 37) % 80)}
              y1={16 + ((index * 19) % 70)}
              x2={12 + ((index * 41) % 80)}
              y2={16 + ((index * 23) % 70)}
              stroke="#a7aca9"
              strokeOpacity={0.09 + (index % 4) * 0.025}
              strokeWidth="0.35"
            />
          ))}
        </g>
        {visibleItems.map((item) => {
          const position = itemPosition(item);
          const active = activeStructure === item.id;
          const labelOnLeft = item.kind === "vessel" || position.x > 66;
          const labelX = labelOnLeft ? Math.max(position.x - 5, 16) : Math.min(position.x + 4, 84);
          const labelY = item.kind === "vessel" ? Math.min(position.y + 6, 88) : Math.max(position.y - 5, 12);
          if (item.kind === "airway") {
            return (
              <g key={item.id} onMouseEnter={() => setActiveStructure(item.id)} onMouseLeave={() => setActiveStructure(null)}>
                <path d="M16 15 C30 23, 70 23, 84 15" stroke={active ? "#fff5c2" : item.color} strokeWidth={active ? 2.2 : 1.5} fill="none" />
                <text x="52" y="19" fill={active ? "#fff5c2" : item.color} fontSize="3.2">airway wall</text>
              </g>
            );
          }
          if (item.kind === "contact") {
            return <circle key={item.id} cx="50" cy="8" r="1.6" fill={item.color} />;
          }
          const shape = itemShape(item);
          const contourPaths = (item.contoursMm ?? [])
            .map((contour) => contourPath(contour))
            .filter((path) => path.length > 0);
          return (
            <g
              key={item.id}
              tabIndex={0}
              className="sector-hotspot"
              data-structure-id={item.id}
              data-shape-aspect={shape.displayAspect.toFixed(2)}
              data-contour-count={contourPaths.length}
              onMouseEnter={() => setActiveStructure(item.id)}
              onMouseLeave={() => setActiveStructure(null)}
              onFocus={() => setActiveStructure(item.id)}
              onBlur={() => setActiveStructure(null)}
            >
              {contourPaths.length > 0 ? (
                contourPaths.map((path, index) => (
                  <path
                    key={`${item.id}-contour-${index}`}
                    d={path}
                    clipPath="url(#fanClip)"
                    fill={path.endsWith(" Z") ? item.color : "none"}
                    fillOpacity={item.kind === "node" ? 0.34 : 0.28}
                    stroke={active ? "#ffffff" : item.color}
                    strokeOpacity={active ? 0.95 : 0.78}
                    strokeWidth={active ? 0.85 : 0.5}
                  />
                ))
              ) : (
                <ellipse
                  cx={position.x}
                  cy={position.y}
                  rx={active ? shape.rx * 1.08 : shape.rx}
                  ry={active ? shape.ry * 1.08 : shape.ry}
                  transform={`rotate(${shape.angleDeg.toFixed(1)} ${position.x.toFixed(2)} ${position.y.toFixed(2)})`}
                  clipPath="url(#fanClip)"
                  fill={item.color}
                  fillOpacity={item.kind === "node" ? 0.66 : 0.52}
                  stroke={active ? "#ffffff" : item.color}
                  strokeWidth={active ? 0.9 : 0.35}
                />
              )}
              {item.kind === "vessel" && (
                <line x1={position.x} y1={position.y} x2={labelX} y2={labelY - 1.2} stroke={item.color} strokeOpacity="0.7" strokeWidth="0.35" />
              )}
              <text x={labelX} y={labelY} textAnchor={labelOnLeft ? "end" : "start"} fill="#d9f0ee" fontSize="3.15">
                {item.label}
              </text>
            </g>
          );
        })}
        <line x1="94" y1="10" x2="94" y2="91" stroke="#758086" strokeWidth="0.6" />
        {[0, 10, 20, 30, 40].map((tick) => (
          <g key={tick}>
            <line x1="92.2" x2="94" y1={8 + (tick / maxDepth) * 82} y2={8 + (tick / maxDepth) * 82} stroke="#758086" strokeWidth="0.55" />
            <text x="95" y={9 + (tick / maxDepth) * 82} fill="#8d999e" fontSize="2.8">{tick}</text>
          </g>
        ))}
        <text x="13" y="96" fill="#8d999e" fontSize="2.8">caudal</text>
        <text x="76" y="96" fill="#8d999e" fontSize="2.8">cephalic</text>
      </svg>
      <div className="structure-list">
        {visibleItems.filter((item) => item.kind !== "contact").map((item) => (
          <button
            key={item.id}
            type="button"
            className={activeStructure === item.id ? "structure-row active" : "structure-row"}
            onMouseEnter={() => setActiveStructure(item.id)}
            onMouseLeave={() => setActiveStructure(null)}
            onFocus={() => setActiveStructure(item.id)}
            onBlur={() => setActiveStructure(null)}
          >
            <span className="swatch" style={{ backgroundColor: item.color }} />
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function App() {
  const { caseData, assets, error } = useCaseData();
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [lineIndex, setLineIndex] = useState<number | null>(null);
  const [sMm, setSMm] = useState(0);
  const [rollDeg, setRollDeg] = useState(0);
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const [teachingView, setTeachingView] = useState(true);
  const [activeStructure, setActiveStructure] = useState<string | null>(null);
  const [volumeSector, setVolumeSector] = useState<VolumeSectorState>({
    status: "idle",
    queryKey: "",
    labels: []
  });

  const selectedPreset = useMemo(() => {
    if (!caseData?.presets.length) {
      return null;
    }
    return caseData.presets.find((preset) => preset.preset_key === selectedKey) ?? caseData.presets[0];
  }, [caseData, selectedKey]);

  useEffect(() => {
    if (!caseData || selectedKey || !caseData.presets.length) {
      return;
    }
    const first = caseData.presets[0];
    setSelectedKey(first.preset_key);
    setLineIndex(first.line_index);
    setSMm(first.centerline_s_mm);
    setRollDeg(caseData.render_defaults.roll_deg);
  }, [caseData, selectedKey]);

  const activePolyline = useMemo(() => {
    if (!assets?.centerlines.polylines.length || !selectedPreset) {
      return null;
    }
    const resolvedLineIndex = lineIndex ?? selectedPreset.line_index;
    return (
      assets.centerlines.polylines.find((polyline) => polyline.line_index === resolvedLineIndex) ??
      assets.centerlines.polylines.find((polyline) => polyline.line_index === selectedPreset.line_index) ??
      assets.centerlines.polylines[0]
    );
  }, [assets, lineIndex, selectedPreset]);

  const pose = useMemo(() => {
    if (!activePolyline || !selectedPreset) {
      return null;
    }
    return computePose(activePolyline, sMm, rollDeg, selectedPreset.target);
  }, [activePolyline, rollDeg, sMm, selectedPreset]);

  const cameraPose = useMemo(() => {
    if (!activePolyline || !selectedPreset) {
      return null;
    }
    return computePose(activePolyline, sMm, 0, selectedPreset.target);
  }, [activePolyline, sMm, selectedPreset]);

  const sectorQueryKey = useMemo(() => {
    if (!caseData || !selectedPreset || !activePolyline) {
      return "";
    }
    return volumeSectorQueryKey(selectedPreset, activePolyline.line_index, sMm, rollDeg, caseData);
  }, [activePolyline, caseData, rollDeg, sMm, selectedPreset]);

  useEffect(() => {
    if (!caseData || !selectedPreset || !activePolyline || !sectorQueryKey) {
      return;
    }

    const controller = new AbortController();
    const params = new URLSearchParams({
      preset_key: selectedPreset.preset_key,
      line_index: String(activePolyline.line_index),
      s_mm: sMm.toFixed(1),
      roll_deg: rollDeg.toFixed(1),
      max_depth_mm: String(caseData.render_defaults.max_depth_mm),
      sector_angle_deg: String(caseData.render_defaults.sector_angle_deg)
    });
    setVolumeSector((current) => ({
      status: current.queryKey === sectorQueryKey && current.status === "ready" ? "ready" : "loading",
      queryKey: sectorQueryKey,
      labels: current.queryKey === sectorQueryKey ? current.labels : []
    }));

    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/sector-volume?${params.toString()}`, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        const payload = (await response.json()) as VolumeSectorResponse;
        setVolumeSector({
          status: "ready",
          queryKey: sectorQueryKey,
          labels: payload.sector.labels
        });
      } catch {
        if (controller.signal.aborted) {
          return;
        }
        setVolumeSector({
          status: "error",
          queryKey: sectorQueryKey,
          labels: []
        });
      }
    }, 90);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [activePolyline, caseData, rollDeg, sMm, sectorQueryKey, selectedPreset]);

  const sectorItems = useMemo<SectorItem[]>(() => {
    if (!caseData || !assets || !pose || !selectedPreset) {
      return [];
    }
    const maxDepth = caseData.render_defaults.max_depth_mm;
    const sectorAngle = caseData.render_defaults.sector_angle_deg;
    const items: SectorItem[] = [
      {
        id: "airway_wall",
        label: "airway wall",
        kind: "airway",
        color: caseData.color_map.airway ?? "#22c7c9",
        depthMm: 2,
        lateralMm: 0,
        visible: true
      },
      {
        id: "contact_region",
        label: "contact region",
        kind: "contact",
        color: "#f5e166",
        depthMm: 0,
        lateralMm: 0,
        visible: true
      }
    ];

    if (volumeSector.status === "ready" && volumeSector.queryKey === sectorQueryKey) {
      items.push(...volumeSector.labels.map(volumeLabelToSectorItem));
      items.sort((a, b) => {
        const order = { airway: 0, contact: 1, node: 2, vessel: 3 };
        return order[a.kind] - order[b.kind] || a.depthMm - b.depthMm || a.label.localeCompare(b.label);
      });
      return items;
    }

    for (const station of caseData.assets.stations) {
      const stationPoints = assets.stations[station.key]?.points ?? [];
      if (!stationPoints.length) {
        continue;
      }
      const projection = nearestVisiblePoint(stationPoints, pose, maxDepth, sectorAngle);
      if (!projection) {
        continue;
      }
      items.push({
        id: station.key,
        label: station.label.replace(" region", ""),
        kind: "node",
        color: caseData.color_map.lymph_node ?? station.color,
        depthMm: projection.depthMm,
        lateralMm: projection.lateralMm,
        visible: projection.visible
      });
    }

    const selectedLineIndex = lineIndex ?? selectedPreset.line_index;
    const atStationSnap =
      selectedLineIndex === selectedPreset.line_index &&
      Math.abs(sMm - selectedPreset.centerline_s_mm) <= 1;
    if (atStationSnap && !items.some((item) => item.kind === "node")) {
      const nodeProjection = projectToSector(
        selectedPreset.target,
        pose,
        maxDepth,
        sectorAngle,
        SNAP_TARGET_SLAB_HALF_THICKNESS_MM
      );
      if (nodeProjection.visible) {
        items.push({
          id: selectedPreset.station_key,
          label: "lymph node",
          kind: "node",
          color: caseData.color_map.lymph_node ?? "#93c56f",
          ...nodeProjection
        });
      }
    }

    for (const listed of caseData.assets.vessels) {
      const vessel = assets.vessels[listed.key];
      if (!vessel) {
        continue;
      }
      const projection = nearestVisiblePoint(vessel.points, pose, maxDepth, sectorAngle);
      if (!projection) {
        continue;
      }
      items.push({
        id: listed.key,
        label: listed.label,
        kind: "vessel",
        color: listed.color,
        depthMm: projection.depthMm,
        lateralMm: projection.lateralMm,
        visible: projection.visible
      });
    }
    items.sort((a, b) => {
      const order = { airway: 0, contact: 1, node: 2, vessel: 3 };
      return order[a.kind] - order[b.kind] || a.depthMm - b.depthMm || a.label.localeCompare(b.label);
    });
    return items;
  }, [assets, caseData, lineIndex, pose, sectorQueryKey, selectedPreset, sMm, volumeSector]);

  const intersectedStructureIds = useMemo(() => {
    return new Set(
      sectorItems
        .filter((item) => item.visible && (item.kind === "node" || item.kind === "vessel"))
        .map((item) => item.id)
    );
  }, [sectorItems]);

  if (error) {
    return (
      <main className="app-shell centered">
        <section className="load-panel">
          <h1>EBUS Anatomy Correlation</h1>
          <p>{error}</p>
        </section>
      </main>
    );
  }

  if (!caseData || !assets || !selectedPreset || !activePolyline || !pose || !cameraPose) {
    return (
      <main className="app-shell centered">
        <section className="load-panel">
          <h1>EBUS Anatomy Correlation</h1>
          <p>Loading case geometry...</p>
        </section>
      </main>
    );
  }

  const snapToPreset = (preset: WebPreset) => {
    setSelectedKey(preset.preset_key);
    setLineIndex(preset.line_index);
    setSMm(preset.centerline_s_mm);
    setRollDeg(caseData.render_defaults.roll_deg);
    setActiveStructure(preset.station_key);
  };

  const updateLayer = (key: keyof LayerState) => {
    setLayers((current) => ({ ...current, [key]: !current[key] }));
  };

  const sectorSource =
    volumeSector.status === "ready" && volumeSector.queryKey === sectorQueryKey
      ? "volume_masks"
      : "point_cloud_fallback";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">{caseData.case_id}</span>
          <h1>EBUS Anatomy Correlation</h1>
        </div>
        <div className="status-strip">
          <span>Station {formatStation(selectedPreset.station)}</span>
          <span>{selectedPreset.approach}</span>
          <span>{Math.round(sMm)} mm</span>
        </div>
      </header>

      <section className="control-rail" aria-label="Navigation controls">
        <label>
          <span>Station snap</span>
          <select
            value={selectedPreset.preset_key}
            onChange={(event) => {
              const preset = caseData.presets.find((candidate) => candidate.preset_key === event.target.value);
              if (preset) {
                snapToPreset(preset);
              }
            }}
          >
            {caseData.presets.map((preset) => (
              <option key={preset.preset_key} value={preset.preset_key}>
                {preset.label}
              </option>
            ))}
          </select>
        </label>
        <label className="wide-control">
          <span>Advance / retract</span>
          <input
            type="range"
            min={0}
            max={activePolyline.total_length_mm}
            step={0.5}
            value={Math.min(Math.max(sMm, 0), activePolyline.total_length_mm)}
            onChange={(event) => setSMm(Number(event.target.value))}
          />
        </label>
        <label>
          <span>Roll</span>
          <input
            type="range"
            min={-45}
            max={45}
            step={1}
            value={rollDeg}
            onChange={(event) => setRollDeg(Number(event.target.value))}
          />
        </label>
        <div className="layer-toggles" aria-label="Anatomy layers">
          <label>
            <input type="checkbox" checked={teachingView} onChange={() => setTeachingView((current) => !current)} />
            <span>teaching</span>
          </label>
          {(Object.keys(layers) as Array<keyof LayerState>).map((key) => (
            <label key={key}>
              <input type="checkbox" checked={layers[key]} onChange={() => updateLayer(key)} />
              <span>{key}</span>
            </label>
          ))}
        </div>
      </section>

      <div className="workspace">
        <section className="scene-pane" aria-label="External anatomy view">
          <div className="pane-header">
            <div>
              <span className="eyebrow">External anatomy</span>
              <h2>Scope, airway, vessels, and fan</h2>
            </div>
            <button type="button" className="snap-button" onClick={() => snapToPreset(selectedPreset)}>
              Snap
            </button>
          </div>
          <AnatomyScene
            caseData={caseData}
            assets={assets}
            pose={pose}
            cameraPose={cameraPose}
            selectedPreset={selectedPreset}
            layers={layers}
            teachingView={teachingView}
            activeStructure={activeStructure}
            intersectedStructureIds={intersectedStructureIds}
          />
        </section>

        <SectorView
          items={sectorItems}
          source={sectorSource}
          selectedPreset={selectedPreset}
          caseData={caseData}
          activeStructure={activeStructure}
          setActiveStructure={setActiveStructure}
        />
      </div>
    </main>
  );
}

export default App;

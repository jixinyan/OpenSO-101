import html
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.viewer.notebook import scene_to_html

from .bundle import verify_bundle
from .capabilities import instance_mesh
from .catalog import AssetCatalog
from .models import file_digest


def export_preview(bundle: Path, output: Path, *, robot_model: Path | None = None, joint_positions=None):
    if output.exists():
        raise FileExistsError(output)
    spec = verify_bundle(bundle)
    catalog = AssetCatalog(bundle / "assets")
    scene = trimesh.Scene()
    for entity in spec.entities:
        mesh = instance_mesh(catalog, entity.asset_uid, entity.dimensions_m)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(entity.pose.quaternion_wxyz, scalar_first=True).as_matrix()
        transform[:3, 3] = entity.pose.position
        scene.add_geometry(mesh, node_name=entity.entity_id, transform=transform)
    table = trimesh.creation.box(extents=(*spec.table.size_xy_m, spec.table.thickness_m))
    table.apply_translation((*spec.table.center_xy_m, spec.table.top_z_m - spec.table.thickness_m / 2))
    table.visual.face_colors = [181, 159, 125, 255]
    scene.add_geometry(table, node_name="table")
    robot_sha256 = None
    if robot_model is not None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(robot_model.resolve()))
        data = mujoco.MjData(model)
        if joint_positions is not None:
            positions = np.asarray(joint_positions, dtype=float)
            if positions.shape != (model.nq,) or not np.isfinite(positions).all():
                raise ValueError("预览需要完整的 MJCF 关节位置")
            data.qpos[:] = positions
        mujoco.mj_forward(model, data)
        robot_transform = np.eye(4)
        robot_transform[:3, :3] = Rotation.from_quat(spec.robot_base.quaternion_wxyz, scalar_first=True).as_matrix()
        robot_transform[:3, 3] = spec.robot_base.position
        for index in range(model.ngeom):
            if (model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH
                    or model.geom_contype[index] != 0 or model.geom_conaffinity[index] != 0):
                continue
            mesh_id = model.geom_dataid[index]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            first, faces = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
            mesh = trimesh.Trimesh(vertices=model.mesh_vert[start:start + count],
                                   faces=model.mesh_face[first:first + faces], process=False)
            transform = np.eye(4)
            transform[:3, :3] = data.geom_xmat[index].reshape(3, 3)
            transform[:3, 3] = data.geom_xpos[index]
            mesh.visual.face_colors = np.asarray(model.geom_rgba[index] * 255, dtype=np.uint8)
            scene.add_geometry(mesh, node_name=f"robot_geom_{index}", transform=robot_transform @ transform)
        robot_sha256 = file_digest(robot_model)
    viewer = scene_to_html(scene)
    document = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>OpenSO-101 场景预览</title>'
        '<style>body{margin:0;font-family:system-ui;background:#eef3f8}header{padding:16px 24px}'
        'iframe{border:0;width:100%;height:78vh}small{word-break:break-all}button{padding:8px 16px}</style>'
        '<header><h2>' + html.escape(spec.scene_id) + '</h2><p>' + html.escape(spec.task.instruction)
        + '</p><small>场景版本：' + spec.digest() + '</small><p>使用鼠标旋转、移动和缩放视角。'
        '物理、可达性和任务成功使用独立验收报告。</p></header><iframe title="实际场景 mesh" srcdoc="'
        + html.escape(viewer, quote=True) + '"></iframe></html>'
    )
    with output.open("x") as stream:
        stream.write(document)
    report = {"scene_sha256": spec.digest(), "manifest_sha256": file_digest(bundle / "manifest.json"),
              "preview_sha256": file_digest(output), "geometry_nodes": len(scene.geometry),
              "robot_model_sha256": robot_sha256, "source_sha256": file_digest(Path(__file__)),
              "status": "interactive_mesh_preview_created", "physical_task_verified": False}
    with output.with_suffix(".preview.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    return report

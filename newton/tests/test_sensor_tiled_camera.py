# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import os
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sensors.warp_raytrace import RenderOrder
from newton.sensors import SensorTiledCamera


def _cubql_available() -> bool:
    return wp.is_cuda_available() and bool(getattr(wp, "is_cubql_available", lambda: False)())


class TestSensorTiledCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            return
        cls._shared_model = cls._build_scene()

    @staticmethod
    def _build_scene():
        from pxr import Usd, UsdGeom

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane(color=(0.91749084, 0.798277, 0.64443165))

        # SPHERE
        sphere_pos = wp.vec3(0.0, -2.0, 0.5)
        body_sphere = builder.add_body(xform=wp.transform(p=sphere_pos, q=wp.quat_identity()), label="sphere")
        builder.add_shape_sphere(body_sphere, radius=0.5, color=(0.5214758, 0.9868272, 0.79823583))

        # CAPSULE
        capsule_pos = wp.vec3(0.0, 0.0, 0.75)
        body_capsule = builder.add_body(xform=wp.transform(p=capsule_pos, q=wp.quat_identity()), label="capsule")
        builder.add_shape_capsule(body_capsule, radius=0.25, half_height=0.5, color=(0.8951316, 0.9551697, 0.8440772))

        # CYLINDER
        cylinder_pos = wp.vec3(0.0, -4.0, 0.5)
        body_cylinder = builder.add_body(xform=wp.transform(p=cylinder_pos, q=wp.quat_identity()), label="cylinder")
        builder.add_shape_cylinder(
            body_cylinder, radius=0.4, half_height=0.5, color=(0.59499574, 0.99073946, 0.64237005)
        )

        # BOX
        box_pos = wp.vec3(0.0, 2.0, 0.5)
        body_box = builder.add_body(xform=wp.transform(p=box_pos, q=wp.quat_identity()), label="box")
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.5, color=(0.8146366, 0.7905182, 0.79995614))

        # MESH (bunny)
        bunny_filename = os.path.join(os.path.dirname(__file__), "..", "examples", "assets", "bunny.usd")
        assert os.path.exists(bunny_filename), f"File not found: {bunny_filename}"
        usd_stage = Usd.Stage.Open(bunny_filename)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        mesh_pos = wp.vec3(0.0, 4.0, 0.0)
        body_mesh = builder.add_body(xform=wp.transform(p=mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), label="mesh")
        builder.add_shape_mesh(body_mesh, mesh=demo_mesh, color=(0.7676241, 0.99788857, 0.75097305))

        return builder.finalize()

    @staticmethod
    def _build_local_box_scene():
        world_builder = newton.ModelBuilder()
        body_box = world_builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5), q=wp.quat_identity()), label="box"
        )
        world_builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.5, color=(0.8146366, 0.7905182, 0.79995614))
        builder = newton.ModelBuilder()
        builder.add_world(world_builder)
        return builder.finalize()

    def __compare_images(self, test_image: np.ndarray, gold_image: np.ndarray, allowed_difference: float = 0.0):
        self.assertEqual(test_image.dtype, gold_image.dtype, "Images have different data types")
        self.assertEqual(test_image.size, gold_image.size, "Images have different data shapes")

        gold_image = gold_image.reshape(test_image.shape)

        # Promote to a wide type before subtracting: int64 avoids unsigned underflow for
        # integer images, float64 preserves fractional deltas for float (e.g. depth) images.
        wide_dtype = np.int64 if np.issubdtype(test_image.dtype, np.integer) else np.float64
        diff = np.abs(test_image.astype(wide_dtype) - gold_image.astype(wide_dtype))

        divider = 1.0
        if np.issubdtype(test_image.dtype, np.integer):
            divider = np.iinfo(test_image.dtype).max

        percentage_diff = float(np.average(diff)) / divider * 100.0
        self.assertLessEqual(
            percentage_diff,
            allowed_difference,
            f"Images differ more than {allowed_difference:.2f}%, total difference is {percentage_diff:.2f}%",
        )

    def __render_small_scene(
        self, model, config: SensorTiledCamera.RenderConfig | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        width = 96
        height = 72
        camera_count = 1

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]], dtype=wp.transformf
        )

        tiled_camera_sensor = SensorTiledCamera(model=model, config=config)
        camera_rays = tiled_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))
        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)

        tiled_camera_sensor.update(
            model.state(), camera_transforms, camera_rays, color_image=color_image, depth_image=depth_image
        )
        wp.synchronize()
        return color_image.numpy(), depth_image.numpy()

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_golden_image(self):
        model = self._shared_model

        width = 320
        height = 240
        camera_count = 1

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]], dtype=wp.transformf
        )

        tiled_camera_sensor = SensorTiledCamera(model=model)
        tiled_camera_sensor.utils.create_default_light(enable_shadows=True)
        tiled_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

        camera_rays = tiled_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))
        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)

        tiled_camera_sensor.update(
            model.state(), camera_transforms, camera_rays, color_image=color_image, depth_image=depth_image
        )

        golden_color_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_sensor_tiled_camera", "color.npy")
        )
        golden_depth_data = np.load(
            os.path.join(os.path.dirname(__file__), "golden_data", "test_sensor_tiled_camera", "depth.npy")
        )

        self.__compare_images(color_image.numpy(), golden_color_data, allowed_difference=0.1)
        self.__compare_images(depth_image.numpy(), golden_depth_data, allowed_difference=0.1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_output_image_parameters(self):
        model = self._shared_model

        width = 640
        height = 480
        camera_count = 1

        camera_transforms = wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))]], dtype=wp.transformf
        )

        tiled_camera_sensor = SensorTiledCamera(model=model)
        camera_rays = tiled_camera_sensor.utils.compute_pinhole_camera_rays(width, height, math.radians(45.0))

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(
            model.state(), camera_transforms, camera_rays, color_image=color_image, depth_image=depth_image
        )
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(
            model.state(), camera_transforms, camera_rays, color_image=color_image, depth_image=None
        )
        self.assertTrue(np.any(color_image.numpy() != 0), "Color image should contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(
            model.state(), camera_transforms, camera_rays, color_image=None, depth_image=depth_image
        )
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertTrue(np.any(depth_image.numpy() != 0), "Depth image should contain rendered data")

        color_image = tiled_camera_sensor.utils.create_color_image_output(width, height, camera_count)
        depth_image = tiled_camera_sensor.utils.create_depth_image_output(width, height, camera_count)
        tiled_camera_sensor.update(model.state(), camera_transforms, camera_rays, color_image=None, depth_image=None)
        self.assertFalse(np.any(color_image.numpy() != 0), "Color image should NOT contain rendered data")
        self.assertFalse(np.any(depth_image.numpy() != 0), "Depth image should NOT contain rendered data")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_no_global_world_primitives_auto_disable_global_world_path(self):
        model = self._build_local_box_scene()
        tiled_camera_sensor = SensorTiledCamera(model=model)
        render_context = tiled_camera_sensor._SensorTiledCamera__render_context

        self.assertTrue(render_context.config.enable_global_world)
        self.assertFalse(render_context.state.enable_global_world)
        self.assertEqual(render_context.world_count_total, render_context.world_count)
        self.assertEqual(render_context.state.shape_type_mask, 1 << int(newton.GeoType.BOX))

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_global_world_primitives_preserve_global_world_path(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane(color=(0.91749084, 0.798277, 0.64443165))
        model = builder.finalize()

        tiled_camera_sensor = SensorTiledCamera(model=model)
        render_context = tiled_camera_sensor._SensorTiledCamera__render_context

        self.assertTrue(render_context.config.enable_global_world)
        self.assertTrue(render_context.state.enable_global_world)
        self.assertEqual(render_context.world_count_total, render_context.world_count + 1)
        self.assertTrue(render_context.state.shape_type_mask & (1 << int(newton.GeoType.PLANE)))

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_auto_disabled_global_world_matches_explicit_disabled_output(self):
        model = self._build_local_box_scene()

        default_color, default_depth = self.__render_small_scene(model)
        explicit_color, explicit_depth = self.__render_small_scene(
            model, SensorTiledCamera.RenderConfig(enable_global_world=False)
        )

        self.__compare_images(default_color, explicit_color)
        self.__compare_images(default_depth, explicit_depth)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_tiled_render_order_matches_pixel_priority_output(self):
        model = self._shared_model

        default_color, default_depth = self.__render_small_scene(
            model, SensorTiledCamera.RenderConfig(render_order=RenderOrder.PIXEL_PRIORITY)
        )
        tiled_color, tiled_depth = self.__render_small_scene(
            model, SensorTiledCamera.RenderConfig(render_order=RenderOrder.TILED, tile_width=16, tile_height=8)
        )

        self.__compare_images(tiled_color, default_color, allowed_difference=0.01)
        self.__compare_images(tiled_depth, default_depth, allowed_difference=0.01)

    @unittest.skipUnless(_cubql_available(), "Requires CUDA and cuBQL-enabled Warp")
    def test_cubql_render_meshes_match_default_renderer_output(self):
        model = self._shared_model

        default_color, default_depth = self.__render_small_scene(model)
        cubql_color, cubql_depth = self.__render_small_scene(
            model, SensorTiledCamera.RenderConfig(mesh_bvh_constructor="cubql")
        )

        self.__compare_images(cubql_color, default_color, allowed_difference=0.01)
        self.__compare_images(cubql_depth, default_depth, allowed_difference=0.01)

    @unittest.skipUnless(_cubql_available(), "Requires CUDA and cuBQL-enabled Warp")
    def test_cubql_render_meshes_do_not_replace_physics_mesh_handles(self):
        model = self._shared_model
        sensor = SensorTiledCamera(model, config=SensorTiledCamera.RenderConfig(mesh_bvh_constructor="cubql"))
        render_context = sensor._SensorTiledCamera__render_context

        physics_source_ids = model.shape_source_ptr.numpy()
        render_source_ids = render_context.shape_source_ptr.numpy()
        mesh_indices = [i for i, shape in enumerate(model.shape_source) if isinstance(shape, newton.Mesh)]

        self.assertTrue(mesh_indices, "Test scene must contain at least one mesh shape")
        for index in mesh_indices:
            self.assertNotEqual(render_source_ids[index], physics_source_ids[index])

        non_mesh_indices = [i for i in range(len(model.shape_source)) if i not in mesh_indices]
        for index in non_mesh_indices:
            self.assertEqual(render_source_ids[index], physics_source_ids[index])


if __name__ == "__main__":
    unittest.main()

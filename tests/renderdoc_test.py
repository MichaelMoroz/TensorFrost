import unittest

import TensorFrost as tf


class RenderDocBindingTest(unittest.TestCase):
    def test_renderdoc_functions_exist(self):
        self.assertTrue(hasattr(tf, "renderdoc_start_capture"))
        self.assertTrue(hasattr(tf, "renderdoc_end_capture"))
        self.assertTrue(hasattr(tf, "renderdoc_is_available"))

    def test_renderdoc_capture_calls(self):
        # Calls shouldn't raise even when RenderDoc isn't attached.
        tf.renderdoc_start_capture()
        path = tf.renderdoc_end_capture()
        self.assertIsInstance(path, str)
        path = tf.renderdoc_end_capture(launch_replay_ui=False)
        self.assertIsInstance(path, str)

    def test_renderdoc_available_returns_bool(self):
        self.assertIsInstance(tf.renderdoc_is_available(), bool)


if __name__ == "__main__":
    unittest.main()

<!DOCTYPE html>
	<head>

		<title>PyntCloud</title>
		<meta charset="utf-8">
			<meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
				<style>
					body {{
						color: #050505;
	font-family: Monospace;
	font-size: 13px;
	text-align: center;
	background-color: #ffffff;
	margin: 0px;
	overflow: hidden;
	align: 'center';
}}
#logo_container {{
						position: absolute;
						top: 0px;
						width: 100 %;
					}}
					.logo {{
						max-width: 20%;
}}
</style>

</head>
			<body>
				<h1> {title} </h1>
				<div id="container">
				</div>

				<script src="http://threejs.org/build/three.js"></script>
				<script src="http://threejs.org/examples/js/WebGL.js"></script>
				<script src="http://threejs.org/examples/js/controls/OrbitControls.js"></script>
				<script src="http://threejs.org/examples/js/libs/stats.min.js"></script>

				<script>

	var container, stats;
	var camera, scene, renderer;
	var points;

	init();
	animate();

	function init() {{

						var camera_x = {camera_x};
						var camera_y = {camera_y};
						var camera_z = {camera_z};

						var look_x = {look_x};
						var look_y = {look_y};
						var look_z = {look_z};

						var positions = new Float32Array({positions});

						var colors = new Float32Array({colors});

						var points_size = {points_size};

						var axis_size = {axis_size};

						container = document.getElementById('container');

						scene = new THREE.Scene();
						scene.background = new THREE.Color(0xffffff)
						camera = new THREE.PerspectiveCamera(90, window.innerWidth / window.innerHeight, 0.1, 1000);
						camera.position.x = camera_x;
						camera.position.y = camera_y;
						camera.position.z = camera_z;
						camera.up = new THREE.Vector3(0, 0, 1);

						if(axis_size > 0) {{
								var axisHelper = new THREE.AxisHelper(axis_size);
								scene.add(axisHelper);
							}}

		var geometry = new THREE.BufferGeometry();
						geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
						geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
						geometry.computeBoundingSphere();

						var material = new THREE.PointsMaterial({{ size: points_size, vertexColors: THREE.VertexColors }} );

		points = new THREE.Points( geometry, material );
		scene.add( points );


		renderer = new THREE.WebGLRenderer( {{ antialias: false, alpha: true }} );
		renderer.setPixelRatio( window.devicePixelRatio );
		renderer.setSize( window.innerWidth, window.innerHeight );
		controls = new THREE.OrbitControls( camera, renderer.domElement );
		controls.target.copy( new THREE.Vector3(look_x, look_y, look_z) );
		camera.lookAt( new THREE.Vector3(look_x, look_y, look_z));

		container.appendChild( renderer.domElement );

		window.addEventListener( 'resize', onWindowResize, false );
	}}

	function onWindowResize() {{
						camera.aspect = window.innerWidth / window.innerHeight;
						camera.updateProjectionMatrix();
						renderer.setSize(window.innerWidth, window.innerHeight);
					}}

					function animate() {{
						requestAnimationFrame(animate);
		render();
					}}

					function render() {{
						renderer.render(scene, camera);
					}}
				</script>

			</body>
</html>
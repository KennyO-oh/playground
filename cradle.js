// Set simulation parameters
const numBalls = 7, 
      ballRadius = 0.5, 
      ballSpacing = 2 * ballRadius * 1.05, 
      stringLength = 5;
const g = 9.8, dt = 0.01, initialAngle = 1.0, damping = 0.01, restitution = 0.96;
let activeCount = 1, side = "left", collisionTriggered = false;
// Calculate initial angular velocity using energy conservation (pendulum formula)
let transferSpeed = Math.sqrt(2 * g * stringLength * (1 - Math.cos(initialAngle))) / stringLength;
const angles = new Array(numBalls).fill(0), 
      angularVelocities = new Array(numBalls).fill(0);

// Set up the scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, stringLength * 0.8, 12);
camera.lookAt(0, stringLength, 0);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
directionalLight.position.set(5, 10, 7.5);
scene.add(directionalLight);
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
hemiLight.position.set(0, 20, 0);
scene.add(hemiLight);

// Materials for the cradle parts
const frameMaterial = new THREE.MeshPhongMaterial({ color: 0x222222, specular: 0x555555, shininess: 50 });
const ballMaterial = new THREE.MeshPhongMaterial({ color: 0x0077ff, specular: 0xffffff, shininess: 100 });
const stringMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });

// Group to hold the cradle elements
const cradleGroup = new THREE.Group();
scene.add(cradleGroup);
const liftOffset = 2;  // Vertical offset to "lift" the entire cradle

// Create the support bar for the cradle
let supportWidth = numBalls * ballSpacing + 1;
const supportGeometry = new THREE.BoxGeometry(supportWidth, 0.2, 0.2);
const support = new THREE.Mesh(supportGeometry, frameMaterial);
support.position.set(0, stringLength, 0);
cradleGroup.add(support);

// Arrays to hold ball, string, and pivot objects
const balls = [], strings = [], pivots = [];

/**
 * Update pivot positions along the support bar and adjust support width.
 */
function updatePivots() {
  pivots.length = 0;
  for (let i = 0; i < numBalls; i++) {
    const x = (i - (numBalls - 1) / 2) * ballSpacing;
    pivots.push(new THREE.Vector3(x, stringLength, 0));
  }
  supportWidth = numBalls * ballSpacing + 1;
  // Scale support mesh to match the new width
  support.scale.set(supportWidth / supportGeometry.parameters.width, 1, 1);
}
updatePivots();

// Create the balls and their connecting strings based on pivot positions
for (let i = 0; i < numBalls; i++) {
  const pivot = pivots[i];
  // Create ball
  const sphereGeometry = new THREE.SphereGeometry(ballRadius, 32, 32);
  const ball = new THREE.Mesh(sphereGeometry, ballMaterial);
  ball.position.set(pivot.x, pivot.y - stringLength, pivot.z - 0.2);
  cradleGroup.add(ball);
  balls.push(ball);
  // Create string using a thin cylinder between pivot and ball
  const stringGeometry = new THREE.CylinderGeometry(0.03, 0.03, stringLength, 8);
  const stringMesh = new THREE.Mesh(stringGeometry, stringMaterial);
  // Set string position and orientation based on ball and pivot positions
  const midPoint = new THREE.Vector3().addVectors(pivot, ball.position).multiplyScalar(0.5);
  stringMesh.position.copy(midPoint);
  const delta = new THREE.Vector3().subVectors(ball.position, pivot);
  const axis = new THREE.Vector3(0, 1, 0).cross(delta).normalize();
  const angleVal = Math.acos(delta.clone().normalize().dot(new THREE.Vector3(0, 1, 0)));
  stringMesh.setRotationFromAxisAngle(axis, angleVal);
  cradleGroup.add(stringMesh);
  strings.push(stringMesh);
}

/**
 * Reset the simulation state.
 * @param {number} n - The number of balls to set in motion.
 */
function resetSimulation(n) {
  activeCount = n;
  side = "left";
  collisionTriggered = false;
  transferSpeed = Math.sqrt(2 * g * stringLength * (1 - Math.cos(initialAngle))) / stringLength;
  updatePivots();
  // Reset all balls to their rest positions with corresponding strings
  for (let i = 0; i < numBalls; i++) {
    angles[i] = 0;
    angularVelocities[i] = 0;
    const pivot = pivots[i];
    balls[i].position.set(pivot.x, pivot.y - stringLength, pivot.z - 0.2);
    const midPoint = new THREE.Vector3().addVectors(pivot, balls[i].position).multiplyScalar(0.5);
    strings[i].position.copy(midPoint);
    const delta = new THREE.Vector3().subVectors(balls[i].position, pivot);
    const axis = new THREE.Vector3(0, 1, 0).cross(delta).normalize();
    const angleVal = Math.acos(delta.clone().normalize().dot(new THREE.Vector3(0, 1, 0)));
    strings[i].setRotationFromAxisAngle(axis, angleVal);
  }
  // Displace the first 'activeCount' balls to initiate the swing
  for (let i = 0; i < activeCount; i++) {
    angles[i] = -initialAngle;
    const pivot = pivots[i];
    const x = pivot.x + stringLength * Math.sin(angles[i]);
    const y = pivot.y - stringLength * Math.cos(angles[i]);
    balls[i].position.set(x, y, pivot.z - 0.2);
    const midPoint = new THREE.Vector3().addVectors(pivot, balls[i].position).multiplyScalar(0.5);
    strings[i].position.copy(midPoint);
    const delta = new THREE.Vector3().subVectors(balls[i].position, pivot);
    const axis = new THREE.Vector3(0, 1, 0).cross(delta).normalize();
    const angleVal = Math.acos(delta.clone().normalize().dot(new THREE.Vector3(0, 1, 0)));
    strings[i].setRotationFromAxisAngle(axis, angleVal);
  }
}

// Attach resetSimulation to button events
document.getElementById('oneBall').addEventListener('click', () => resetSimulation(1));
document.getElementById('twoBalls').addEventListener('click', () => resetSimulation(2));
document.getElementById('threeBalls').addEventListener('click', () => resetSimulation(3));
document.getElementById('fourBalls').addEventListener('click', () => resetSimulation(4));

// Translate the cradle to be more vertically centered
cradleGroup.position.y = 2;

// Animate the simulation
function animate() {
  requestAnimationFrame(animate);
  if (side === "left") {
    // Update left-side (active) balls
    for (let i = 0; i < activeCount; i++) {
      const a = -(g / stringLength) * Math.sin(angles[i]);
      angularVelocities[i] += a * dt;
      angularVelocities[i] *= (1 - damping * dt);
      angles[i] += angularVelocities[i] * dt;
      const pivot = pivots[i];
      const x = pivot.x + stringLength * Math.sin(angles[i]);
      const y = pivot.y - stringLength * Math.cos(angles[i]);
      balls[i].position.set(x, y, pivot.z - 0.2);
      const midPoint = new THREE.Vector3().addVectors(pivot, balls[i].position).multiplyScalar(0.5);
      strings[i].position.copy(midPoint);
      const delta = new THREE.Vector3().subVectors(balls[i].position, pivot);
      const axis = new THREE.Vector3(0, 1, 0).cross(delta).normalize();
      const angleVal = Math.acos(delta.clone().normalize().dot(new THREE.Vector3(0, 1, 0)));
      strings[i].setRotationFromAxisAngle(axis, angleVal);
    }
    // Detect collision when the last active ball nears vertical (small angle) and is moving right
    if (!collisionTriggered && Math.abs(angles[activeCount - 1]) < 0.02 && angularVelocities[activeCount - 1] > 0) {
      for (let i = 0; i < activeCount; i++) { 
        angles[i] = 0; 
        angularVelocities[i] = 0; 
      }
      for (let i = numBalls - activeCount; i < numBalls; i++) { 
        angularVelocities[i] = transferSpeed; 
      }
      transferSpeed *= restitution;
      side = "right";
      collisionTriggered = true;
    }
  } else if (side === "right") {
    // Update right-side (active) balls
    for (let i = numBalls - activeCount; i < numBalls; i++) {
      const a = -(g / stringLength) * Math.sin(angles[i]);
      angularVelocities[i] += a * dt;
      angularVelocities[i] *= (1 - damping * dt);
      angles[i] += angularVelocities[i] * dt;
      const pivot = pivots[i];
      const x = pivot.x + stringLength * Math.sin(angles[i]);
      const y = pivot.y - stringLength * Math.cos(angles[i]);
      balls[i].position.set(x, y, pivot.z - 0.2);
      const midPoint = new THREE.Vector3().addVectors(pivot, balls[i].position).multiplyScalar(0.5);
      strings[i].position.copy(midPoint);
      const delta = new THREE.Vector3().subVectors(balls[i].position, pivot);
      const axis = new THREE.Vector3(0, 1, 0).cross(delta).normalize();
      const angleVal = Math.acos(delta.clone().normalize().dot(new THREE.Vector3(0, 1, 0)));
      strings[i].setRotationFromAxisAngle(axis, angleVal);
    }
    // Collision detection for right-side active balls (moving left)
    if (!collisionTriggered && Math.abs(angles[numBalls - activeCount]) < 0.02 && angularVelocities[numBalls - activeCount] < 0) {
      for (let i = numBalls - activeCount; i < numBalls; i++) { 
        angles[i] = 0; 
        angularVelocities[i] = 0; 
      }
      for (let i = 0; i < activeCount; i++) { 
        angularVelocities[i] = -transferSpeed; 
      }
      transferSpeed *= restitution;
      side = "left";
      collisionTriggered = true;
    }
  }
  // Reset non-active balls to rest positions
  for (let i = 0; i < numBalls; i++) {
    if ((side === "left" && i < activeCount) || (side === "right" && i >= numBalls - activeCount)) continue;
    angles[i] = 0; angularVelocities[i] = 0;
    const pivot = pivots[i];
    balls[i].position.set(pivot.x, pivot.y - stringLength, pivot.z - 0.2);
    const midPoint = new THREE.Vector3().addVectors(pivot, balls[i].position).multiplyScalar(0.5);
    strings[i].position.copy(midPoint);
    const delta = new THREE.Vector3().subVectors(balls[i].position, pivot);
    const axis = new THREE.Vector3(0, 1, 0).cross(delta).normalize();
    const angleVal = Math.acos(delta.clone().normalize().dot(new THREE.Vector3(0, 1, 0)));
    strings[i].setRotationFromAxisAngle(axis, angleVal);
  }
  // Re-enable collision detection after the active ball swings past a threshold angle
  if (side === "left" && Math.abs(angles[activeCount - 1]) > 0.05) {
    collisionTriggered = false;
  } else if (side === "right" && Math.abs(angles[numBalls - activeCount]) > 0.05) {
    collisionTriggered = false;
  }
  renderer.render(scene, camera);
}
animate();
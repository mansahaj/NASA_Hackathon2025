import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, useTexture } from "@react-three/drei";

function SunMaterial({ textureUrl }) {
  const map = useTexture(textureUrl);
  return (
    <meshStandardMaterial
      map={map}
      emissive={"#ffcc33"}
      emissiveIntensity={1.0}
      color={"#ffaa00"}
    />
  );
}

function Sun({ textureUrl }) {
  return (
    <mesh>
      <sphereGeometry args={[1.2, 32, 32]} />
      {textureUrl ? (
        <SunMaterial textureUrl={textureUrl} />
      ) : (
        <meshStandardMaterial
          emissive={"#ffcc33"}
          emissiveIntensity={1.2}
          color={"#ffaa00"}
        />
      )}
    </mesh>
  );
}

function PlanetMaterial({ textureUrl, color }) {
  const map = useTexture(textureUrl);
  return <meshStandardMaterial map={map} color={color || "white"} />;
}

function Planet({
  distance = 3,
  size = 0.3,
  color = "#77aaff",
  speed = 0.5,
  textureUrl,
}) {
  const [theta] = React.useState(Math.random() * Math.PI * 2);
  const ref = React.useRef();
  useAnimation(ref, distance, speed, theta);
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[size, 32, 32]} />
      {textureUrl ? (
        <PlanetMaterial textureUrl={textureUrl} color={color} />
      ) : (
        <meshStandardMaterial color={color} />
      )}
    </mesh>
  );
}

function useAnimation(ref, distance, speed, initialTheta) {
  React.useEffect(() => {
    let frame;
    const animate = () => {
      const time = performance.now() / 1000;
      const angle = initialTheta + time * speed;
      if (ref.current) {
        ref.current.position.x = Math.cos(angle) * distance;
        ref.current.position.z = Math.sin(angle) * distance;
      }
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [ref, distance, speed, initialTheta]);
}

export default function ExoplanetInfo() {
  return (
    <div className="page">
      <div className="container">
        <h2>What are Exoplanets?</h2>
        <p>
          Exoplanets are planets that orbit stars beyond our Sun. Thousands have
          been discovered using methods like transit photometry, where a planet
          passes in front of its star causing a tiny dip in brightness.
          Exploring exoplanets helps us understand how planetary systems form
          and whether some worlds might be habitable.
        </p>
      </div>
      <div className="canvas-wrap">
        <Canvas camera={{ position: [0, 3, 8], fov: 60 }}>
          <ambientLight intensity={1.3} />
          <pointLight position={[0, 0, 0]} intensity={2} color={"#ffd27d"} />
          <Suspense fallback={null}>
            <Stars
              radius={50}
              depth={20}
              count={2000}
              factor={4}
              fade
              speed={1}
            />
            <Sun textureUrl={"/textures/sun.jpg"} />
            <Planet
              distance={3}
              size={0.35}
              color={"#7fc8ff"}
              speed={0.6}
              textureUrl={"/textures/earth.jpg"}
            />
            {/* <Planet
              distance={4.5}
              size={0.25}
              color={"#a0ffa0"}
              speed={0.45}
              // textureUrl={"/textures/planet2.jpg"}
            />
            <Planet
              distance={6}
              size={0.6}
              color={"#c8a57f"}
              speed={0.25}
              // textureUrl={"/textures/planet3.jpg"}
            /> */}
          </Suspense>
          <OrbitControls enableDamping dampingFactor={0.05} />
        </Canvas>
      </div>
      <div className="container">
        <h3>How do we detect them?</h3>
        <ul className="bullets">
          <li>
            <strong>Transit</strong>: Measure dips in starlight when a planet
            passes in front.
          </li>
          <li>
            <strong>Radial Velocity</strong>: Detect star wobble from a planet’s
            gravity.
          </li>
          <li>
            <strong>Direct Imaging</strong>: Capture faint light from planets
            directly.
          </li>
        </ul>
      </div>
    </div>
  );
}

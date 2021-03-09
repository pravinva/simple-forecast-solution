// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { HTMLAttributes, useState, useEffect } from "react";
import styles from "./Sidebar.module.scss";
import { Logo } from "../../common/panel/logo/Logo";
import classNames from "classnames";
import { Link } from "../../common/link/Link";
import { UserService, UserAccountInfoResponse, UserRole } from "../../../services/user";
import { AdminSidebar } from "./admin/AdminSidebar";
import { UserSidebar } from "./user/UserSidebar";

const userService = new UserService();

export const Sidebar: React.FC<HTMLAttributes<HTMLDivElement>> = ({ children, ...props }) => {
  const [role, setRole] = useState<UserRole>();

  useEffect(() => {
    loadRole();
  }, []);

  const loadRole = async () => {
    setRole(await userService.getUserRole());
  };

  return (
    <section {...props} className={classNames(styles.sidebar, props.className)}>
      <header>
        <Link href="/authenticated">
          <Logo />
        </Link>
      </header>
      <div className={styles.body}>
        {role === UserRole.Standard && <UserSidebar />}
        {role === UserRole.Admin && <AdminSidebar />}
      </div>
    </section>
  );
};
